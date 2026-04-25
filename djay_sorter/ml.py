"""ML transition model — training data, model training, ML cost matrix."""

import logging
import os
import pickle
import random
import re
import shutil
import struct
from urllib.parse import unquote

import numpy as np

from .constants import (
    APPLE_EPOCH_TS,
    CAMELOT,
    MIN_SESSION_TRACKS,
    MODEL_FILE,
    RECORDINGS_DIR,
)

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def _camelot_distance(c1, c2):
    """Camelot wheel distance between two (number, letter) key tuples."""
    n1, l1 = c1
    n2, l2 = c2
    if n1 == n2 and l1 == l2:
        return 0.0
    if n1 == n2:
        return 0.5
    num_diff = min(abs(n1 - n2), 12 - abs(n1 - n2))
    return num_diff + (0.0 if l1 == l2 else 0.5)


# ---------------------------------------------------------------------------
# Session parsing
# ---------------------------------------------------------------------------

def _parse_session_items(db):
    """Return a dict: session_uuid → list of (start_time_float, title_id) sorted by start_time."""
    items_by_session = {}
    for data in db.execute(
        "SELECT data FROM database2 WHERE collection='historySessionItems'"
    ):
        data = data[0]
        parts = re.findall(b'\x08([^\x00]+)\x00', data)
        strings = [p.decode('utf-8', errors='replace') for p in parts]
        if len(strings) < 7 or strings[0] != 'ADCHistorySessionItem':
            continue
        session_uuid = strings[3]
        title_id     = strings[6]

        start_time = None
        for i in range(len(data) - 8):
            val = struct.unpack_from('<d', data, i)[0]
            if 5e8 < val < 2e9:
                start_time = val
                break
        if start_time is None:
            continue

        items_by_session.setdefault(session_uuid, []).append((start_time, title_id))

    for session_uuid in items_by_session:
        items_by_session[session_uuid].sort(key=lambda x: x[0])

    return items_by_session


def _parse_all_items_flat(db):
    """Return all session items as a flat list of (unix_timestamp, title_id)."""
    items = []
    for data, in db.execute(
        "SELECT data FROM database2 WHERE collection='historySessionItems'"
    ):
        parts = re.findall(b'\x08([^\x00]+)\x00', data)
        strings = [p.decode('utf-8', errors='replace') for p in parts]
        if len(strings) < 7 or strings[0] != 'ADCHistorySessionItem':
            continue
        title_id = strings[6]
        for i in range(len(data) - 8):
            val = struct.unpack_from('<d', data, i)[0]
            if 5e8 < val < 2e9:
                items.append((APPLE_EPOCH_TS + val, title_id))
                break
    return items


def _recording_track_lists(db):
    """Match djay Pro session items to recording files by timestamp."""
    if not os.path.isdir(RECORDINGS_DIR):
        return []

    if not shutil.which('ffprobe'):
        logger.warning(
            "ffprobe not found on PATH — skipping recording-based training data. "
            "Install with 'brew install ffmpeg'."
        )
        return []

    all_items = _parse_all_items_flat(db)
    if not all_items:
        return []

    import subprocess
    track_lists = []
    for fname in os.listdir(RECORDINGS_DIR):
        if not fname.endswith('.m4a') or fname.startswith('.'):
            continue
        path = os.path.join(RECORDINGS_DIR, fname)
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True
        )
        if not result.stdout.strip():
            continue
        dur = float(result.stdout.strip())
        if dur < MIN_SESSION_TRACKS * 30:
            continue

        rec_end   = os.path.getmtime(path)
        rec_start = rec_end - dur

        matched = sorted(
            [(ts, tid) for ts, tid in all_items if rec_start <= ts <= rec_end],
            key=lambda x: x[0]
        )
        if len(matched) >= MIN_SESSION_TRACKS:
            track_lists.append([tid for _, tid in matched])

    return track_lists


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def build_training_data(db, feature_cache, use_recordings=True):
    """Extract (feature_vector, label) pairs from djay Pro set history."""
    path_by_title_id = {}
    for key, data in db.execute(
        "SELECT key, data FROM database2 WHERE collection='localMediaItemLocations'"
    ):
        urls = re.findall(b'file:///[^\x00\x0a]+', data)
        if urls:
            path = unquote(urls[0].decode().replace('file://', ''))
            path_by_title_id[key] = path

    def features_for(title_id):
        path = path_by_title_id.get(title_id)
        if not path:
            return None
        entry = feature_cache.get(path)
        if not entry or 'spectral_flux' not in entry:
            return None
        return entry

    def feature_vector(a, b):
        bpm_a, bpm_b = a.get('tempo', 0) or 0, b.get('tempo', 0) or 0
        bpm_diff = min(
            abs(bpm_a - bpm_b),
            abs(bpm_a - bpm_b * 2),
            abs(bpm_a * 2 - bpm_b),
        )
        camelot_a = CAMELOT.get((a['key'], a['mode']), (0, '?'))
        camelot_b = CAMELOT.get((b['key'], b['mode']), (0, '?'))
        key_dist  = _camelot_distance(camelot_a, camelot_b)
        return [
            bpm_diff,
            key_dist,
            abs(a['spectral_flux']      - b['spectral_flux']),
            abs(a['spectral_centroid']  - b['spectral_centroid']),
            abs(a['onset_density']      - b['onset_density']),
        ]

    title_id_lists = []
    source_label = 'session history'

    if use_recordings:
        rec_lists = _recording_track_lists(db)
        if rec_lists:
            title_id_lists = rec_lists
            source_label = f'recordings ({len(rec_lists)} files)'

    if not title_id_lists:
        items_by_session = _parse_session_items(db)
        title_id_lists = [
            [tid for _, tid in items]
            for items in items_by_session.values()
            if len(items) >= MIN_SESSION_TRACKS
        ]

    logger.info('Training source: %s', source_label)

    X, y = [], []

    for title_ids in title_id_lists:
        resolved = []
        for title_id in title_ids:
            feat = features_for(title_id)
            if feat:
                resolved.append(feat)

        if len(resolved) < 4:
            continue

        for i in range(len(resolved) - 1):
            X.append(feature_vector(resolved[i], resolved[i + 1]))
            y.append(1)

        n_pos = len(resolved) - 1
        attempts = 0
        n_neg = 0
        while n_neg < n_pos and attempts < n_pos * 10:
            attempts += 1
            i, j = sorted(random.sample(range(len(resolved)), 2))
            if j - i <= 1:
                continue
            X.append(feature_vector(resolved[i], resolved[j]))
            y.append(0)
            n_neg += 1

    return np.array(X, dtype=float), np.array(y, dtype=int)


# ---------------------------------------------------------------------------
# Model training / loading
# ---------------------------------------------------------------------------

def train_transition_model(db, feature_cache):
    """Train a RandomForestClassifier on set history and save it to MODEL_FILE."""
    if not SKLEARN_AVAILABLE:
        logger.warning('scikit-learn not installed — skipping ML model. Run: pip install scikit-learn')
        return None

    logger.info('Building training data…')
    X, y = build_training_data(db, feature_cache)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    logger.info('%d positive pairs, %d negative pairs', n_pos, n_neg)

    MIN_PAIRS = 200
    MIN_AUC   = 0.65

    if len(X) < MIN_PAIRS:
        logger.warning('Not enough training data (%d pairs, need ≥%d).', len(X), MIN_PAIRS)
        logger.warning('Run the sorter on more playlists to populate the feature cache, then retrain.')
        return None

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    auc = scores.mean()
    logger.info('Cross-val AUC: %.3f ± %.3f', auc, scores.std())

    if auc < MIN_AUC:
        logger.warning('AUC %.3f is below threshold (%.2f) — model is not better than chance.', auc, MIN_AUC)
        logger.warning('Sticking with heuristic cost function.')
        return None

    model.fit(X, y)

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    logger.info('Model saved → %s', MODEL_FILE)

    importances = model.feature_importances_
    feat_names = ['bpm_diff', 'key_dist', 'flux_diff', 'centroid_diff', 'onset_diff']
    for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
        logger.info('  %-16s %.3f', name, imp)

    return model


def load_transition_model():
    """Load a previously trained model from disk. Returns None if not found."""
    if not SKLEARN_AVAILABLE or not os.path.exists(MODEL_FILE):
        return None
    with open(MODEL_FILE, 'rb') as f:
        return pickle.load(f)


def ml_transition_cost(a, b, model):
    """Use the ML model's transition probability as cost (lower = better)."""
    bpm_a = a.get('tempo', 0) or 0
    bpm_b = b.get('tempo', 0) or 0
    bpm_diff = min(abs(bpm_a - bpm_b), abs(bpm_a - bpm_b * 2), abs(bpm_a * 2 - bpm_b))
    key_dist  = _camelot_distance(a['camelot'], b['camelot'])
    vec = [[
        bpm_diff,
        key_dist,
        abs(a.get('spectral_flux', 0)      - b.get('spectral_flux', 0)),
        abs(a.get('spectral_centroid', 0)  - b.get('spectral_centroid', 0)),
        abs(a.get('onset_density', 0)      - b.get('onset_density', 0)),
    ]]
    prob_good = model.predict_proba(vec)[0][1]
    return 1.0 - prob_good


def _build_ml_cost_matrix(tracks, model):
    """Vectorised ML cost matrix — calls predict_proba once for all N×N pairs."""
    n = len(tracks)
    matrix = [[0.0] * n for _ in range(n)]
    if n < 2:
        return matrix

    tempos = np.array([t['tempo'] for t in tracks])
    fluxes = np.array([t.get('spectral_flux', 0) for t in tracks])
    centroids = np.array([t.get('spectral_centroid', 0) for t in tracks])
    onset_densities = np.array([t.get('onset_density', 0) for t in tracks])
    camelots = [t['camelot'] for t in tracks]

    pairs_i, pairs_j = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
    tempos_i, tempos_j = tempos[list(pairs_i)], tempos[list(pairs_j)]

    bpm_diffs = np.minimum(
        np.abs(tempos_i - tempos_j),
        np.minimum(np.abs(tempos_i - tempos_j * 2), np.abs(tempos_i * 2 - tempos_j)),
    )

    key_dists = np.array([
        _camelot_distance(camelots[i], camelots[j])
        for i, j in zip(pairs_i, pairs_j)
    ])

    feature_matrix = np.column_stack([
        bpm_diffs,
        key_dists,
        np.abs(fluxes[list(pairs_i)] - fluxes[list(pairs_j)]),
        np.abs(centroids[list(pairs_i)] - centroids[list(pairs_j)]),
        np.abs(onset_densities[list(pairs_i)] - onset_densities[list(pairs_j)]),
    ])

    probs = model.predict_proba(feature_matrix)[:, 1]
    costs = 1.0 - probs

    for (i, j), cost in zip(zip(pairs_i, pairs_j), costs):
        matrix[i][j] = float(cost)

    return matrix
