#!/usr/bin/env python3
"""
DJ Playlist Optimiser — djay Pro edition

Reads playlists and BPM directly from djay Pro's database.
Detects musical key using librosa (more accurate than djay Pro's key analysis).
Sorts by Camelot key compatibility + minimal BPM jumps.
Writes sorted playlist back to Apple Music (visible in djay Pro).

Usage:
    python djay_sorter.py
"""

import argparse
import concurrent.futures
import datetime
import json
import logging
import math
import os
import pickle
import random
import re
import shutil
import struct
import sqlite3
import subprocess
import sys
import threading
import uuid
import warnings
from urllib.parse import unquote

logger = logging.getLogger(__name__)

import librosa
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Suppress librosa's verbose deprecation/numba warnings without silencing everything
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')

DJAY_DB            = os.path.expanduser('~/Music/djay/djay Media Library.djayMediaLibrary/MediaLibrary.db')
CACHE_FILE         = os.path.expanduser('~/.dj_key_cache.json')
MODEL_FILE         = os.path.expanduser('~/.dj_transition_model.pkl')
AUDIO_EXTS         = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.aiff', '.aif', '.mp4'}
MIN_SESSION_TRACKS = 20   # sessions shorter than this are treated as testing noise

# ---------------------------------------------------------------------------
# Camelot wheel (for librosa output: Spotify-style key 0-11, mode 0/1)
# ---------------------------------------------------------------------------
CAMELOT = {
    (0,  0): (5,  'A'),  (1,  0): (12, 'A'),  (2,  0): (7,  'A'),
    (3,  0): (2,  'A'),  (4,  0): (9,  'A'),  (5,  0): (4,  'A'),
    (6,  0): (11, 'A'),  (7,  0): (6,  'A'),  (8,  0): (1,  'A'),
    (9,  0): (8,  'A'),  (10, 0): (3,  'A'),  (11, 0): (10, 'A'),
    (0,  1): (8,  'B'),  (1,  1): (3,  'B'),  (2,  1): (10, 'B'),
    (3,  1): (5,  'B'),  (4,  1): (12, 'B'),  (5,  1): (7,  'B'),
    (6,  1): (2,  'B'),  (7,  1): (9,  'B'),  (8,  1): (4,  'B'),
    (9,  1): (11, 'B'),  (10, 1): (6,  'B'),  (11, 1): (1,  'B'),
}

_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                   5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                   4.75, 3.98, 2.69, 3.34, 3.17])


# ---------------------------------------------------------------------------
# djay Pro database
# ---------------------------------------------------------------------------

def open_djay_db(custom_path=None):
    path = custom_path or DJAY_DB
    if not os.path.exists(path):
        raise FileNotFoundError(f"djay Pro database not found at {path}")
    db = sqlite3.connect(path)
    db.execute('PRAGMA query_only = ON')
    return db


def list_playlists(db):
    # Single query: join playlist index against relationship table to get track counts,
    # then filter to playlists that actually contain tracks.
    rows = db.execute('''
        SELECT p.rowid, p.name, COUNT(r.src) as track_count
        FROM secondaryIndex_mediaItemPlaylistIndex p
        JOIN relationship_relationship r
            ON r.name = "mediaItemPlaylistItemPlaylist" AND r.dst = p.rowid
        GROUP BY p.rowid, p.name
        HAVING track_count > 0
        ORDER BY p.name
    ''').fetchall()
    return [(rowid, name, count) for rowid, name, count in rows]


def _track_label(media_rowid, fts_titles, path=None):
    title, artist = fts_titles.get(media_rowid, ('', ''))
    if title:
        return f"{artist} – {title}" if artist else title
    if path:
        return os.path.basename(path)
    return f"track #{media_rowid}"


def get_playlist_tracks(db, playlist_rowid):
    # Fetch only the media rowids that belong to this playlist
    rows = db.execute('''
        SELECT r_media.dst, idx.bpm
        FROM relationship_relationship r_item
        JOIN relationship_relationship r_media
            ON r_media.src = r_item.src
            AND r_media.name = "mediaItemPlaylistItemMediaItem"
        LEFT JOIN secondaryIndex_mediaItemIndex idx ON idx.rowid = r_media.dst
        WHERE r_item.name = "mediaItemPlaylistItemPlaylist"
          AND r_item.dst = ?
    ''', (playlist_rowid,)).fetchall()

    if not rows:
        return [], []

    media_rowids = [r[0] for r in rows]
    bpm_by_rowid = {r[0]: r[1] for r in rows}

    # Fetch only the keys for these rowids
    placeholders = ','.join('?' * len(media_rowids))
    rowid_to_key = dict(db.execute(
        f"SELECT rowid, key FROM database2 WHERE collection='mediaItems' AND rowid IN ({placeholders})",
        media_rowids,
    ))

    # Fetch file paths only for these keys
    keys = list(rowid_to_key.values())
    path_by_key = {}
    if keys:
        placeholders_k = ','.join('?' * len(keys))
        for key, data in db.execute(
            f"SELECT key, data FROM database2 WHERE collection='localMediaItemLocations' AND key IN ({placeholders_k})",
            keys,
        ):
            urls = re.findall(b'file:///[^\x00\x0a]+', data)
            if urls:
                path_by_key[key] = unquote(urls[0].decode().replace('file://', ''))

    # Fetch titles only for these rowids
    fts_titles = {}
    for rowid, title, artist in db.execute(
        f"SELECT rowid, c0title, c1artist FROM fts_searchIndex_content WHERE rowid IN ({placeholders})",
        media_rowids,
    ):
        fts_titles[rowid] = (title or '', artist or '')

    tracks = []
    skip_reasons = []
    for media_rowid in media_rowids:
        item_key = rowid_to_key.get(media_rowid)
        if not item_key:
            skip_reasons.append((_track_label(media_rowid, fts_titles), 'no database entry'))
            continue
        path = path_by_key.get(item_key)
        if not path:
            skip_reasons.append((_track_label(media_rowid, fts_titles), 'no file path in database'))
            continue
        if not os.path.exists(path):
            skip_reasons.append((_track_label(media_rowid, fts_titles, path), 'file not found'))
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext not in AUDIO_EXTS:
            skip_reasons.append((_track_label(media_rowid, fts_titles, path), f'unsupported format ({ext})'))
            continue

        title, artist = fts_titles.get(media_rowid, ('', ''))
        if not title:
            title = os.path.splitext(os.path.basename(path))[0]

        tracks.append({
            'name':    title,
            'artist':  artist,
            'path':    path,
            'tempo':   bpm_by_rowid.get(media_rowid) or 0.0,
            '_rowid':  media_rowid,
        })

    if skip_reasons:
        logger.warning('Skipped %d track(s):', len(skip_reasons))
        for label, reason in skip_reasons:
            logger.warning('  • %s — %s', label, reason)

    return tracks, skip_reasons


# ---------------------------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------------------------

def load_key_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_key_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def extract_audio_features(path):
    """Extract key, mode, spectral flux, spectral centroid, onset density, and BPM."""
    y, sr = librosa.load(path, mono=True, duration=120)

    # Key detection via Krumhansl-Schmuckler key profiles
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)
    best_score, best_key, best_mode = -np.inf, 0, 1
    for k in range(12):
        for mode, profile in ((1, _MAJOR), (0, _MINOR)):
            score = float(np.corrcoef(mean_chroma, np.roll(profile, k))[0, 1])
            if score > best_score:
                best_score, best_key, best_mode = score, k, mode

    # Onset envelope — reused for both spectral flux and BPM detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    spectral_flux = float(np.mean(onset_env))

    # BPM via beat tracking (reuses onset envelope, no extra cost)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    detected_bpm = float(np.atleast_1d(tempo)[0])

    # Spectral centroid — brightness proxy (Hz)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # Onset density — transient/busyness proxy (onsets per second)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration = len(y) / sr
    onset_density = len(onset_frames) / duration if duration > 0 else 0.0

    return best_key, best_mode, spectral_flux, centroid, onset_density, detected_bpm


def enrich_with_keys(tracks, progress_cb=None):
    cache = load_key_cache()
    total = len(tracks)

    # Split tracks into cache hits and those needing analysis
    cached_entries = {}   # idx → cache entry dict
    to_analyse = []       # list of (idx, track, mtime)

    for i, t in enumerate(tracks):
        path = t['path']
        mtime = str(os.path.getmtime(path))
        entry = cache.get(path, {})
        needs = (
            entry.get('mtime') != mtime
            or 'spectral_flux' not in entry    # re-analyse old cache entries missing new fields
            or 'detected_bpm' not in entry     # re-analyse entries missing BPM fallback
        )
        if needs:
            to_analyse.append((i, t, mtime))
        else:
            cached_entries[i] = entry

    # Progress counter: cache hits are already "done"
    done = [len(cached_entries)]
    lock = threading.Lock()
    new_cache_entries = {}  # path → entry (merged after all workers finish)
    analysis_results = {}   # idx → (key, mode, flux, centroid, density, bpm)
    failed = set()          # indices whose analysis failed

    def _analyse(args):
        idx, t, mtime = args
        path = t['path']
        logger.info('[%d/%d] Analysing: %s', idx + 1, total, t['name'][:50])
        try:
            feats = extract_audio_features(path)
            analysis_results[idx] = feats
            new_cache_entries[path] = {
                'mtime': mtime, 'key': feats[0], 'mode': feats[1],
                'spectral_flux': feats[2], 'spectral_centroid': feats[3],
                'onset_density': feats[4], 'detected_bpm': feats[5],
            }
        except Exception as e:
            logger.warning('Analysis failed for %s: %s', t['name'], e)
            failed.add(idx)
        finally:
            with lock:
                done[0] += 1
                if progress_cb:
                    progress_cb(done[0] / total, f'Analysing {done[0]}/{total}…')

    # Parallelise I/O-bound audio loading across up to 4 threads
    workers = min(4, len(to_analyse)) if to_analyse else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pool.map(_analyse, to_analyse)

    if new_cache_entries:
        cache.update(new_cache_entries)
        save_key_cache(cache)

    # Rebuild enriched list preserving original track order
    enriched = []
    for i, t in enumerate(tracks):
        if i in failed:
            continue
        if i in cached_entries:
            e = cached_entries[i]
            key, mode = e['key'], e['mode']
            spectral_flux, centroid = e['spectral_flux'], e['spectral_centroid']
            onset_density, detected_bpm = e['onset_density'], e['detected_bpm']
        else:
            key, mode, spectral_flux, centroid, onset_density, detected_bpm = analysis_results[i]

        # Use djay Pro's BPM if available; fall back to librosa detection
        final_tempo = t['tempo'] or detected_bpm
        camelot = CAMELOT.get((key, mode), (0, '?'))
        enriched.append({
            **t, 'tempo': final_tempo, 'key': key, 'mode': mode, 'camelot': camelot,
            'spectral_flux': spectral_flux, 'spectral_centroid': centroid,
            'onset_density': onset_density,
        })

    return enriched


RECORDINGS_DIR = os.path.expanduser('~/Music/djay/Recordings')
APPLE_EPOCH_TS = 978307200  # 2001-01-01 00:00:00 as Unix timestamp


# ---------------------------------------------------------------------------
# ML — build training data from djay Pro set history
# ---------------------------------------------------------------------------

def _parse_session_items(db):
    """Return a dict: session_uuid → list of (start_time_float, title_id) sorted by start_time."""
    # titleID (hex hash) links session items to localMediaItemLocations keys
    items_by_session = {}
    for data in db.execute(
        "SELECT data FROM database2 WHERE collection='historySessionItems'"
    ):
        data = data[0]
        parts = re.findall(b'\x08([^\x00]+)\x00', data)
        strings = [p.decode('utf-8', errors='replace') for p in parts]
        # strings layout: [0]=class [1]=item_uuid [2]='uuid'
        #                 [3]=session_uuid [4]='sessionUUID'
        #                 [5]='ADCMediaItemTitleID' [6]=title_id_hex [7]=title
        #                 [8]='title' [9]=artist [10]='artist' ...
        if len(strings) < 7 or strings[0] != 'ADCHistorySessionItem':
            continue
        session_uuid = strings[3]
        title_id     = strings[6]

        # Parse startTime: an 8-byte little-endian double (NSTimeInterval) found
        # by scanning for plausible Apple-epoch timestamps (> 5×10^8)
        start_time = None
        for i in range(len(data) - 8):
            val = struct.unpack_from('<d', data, i)[0]
            if 5e8 < val < 2e9:
                start_time = val
                break
        if start_time is None:
            continue

        items_by_session.setdefault(session_uuid, []).append((start_time, title_id))

    # Sort each session's items by start_time
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
    """Match djay Pro session items to recording files by timestamp.

    For each recording whose duration is ≥ MIN_SESSION_TRACKS * 30s (roughly),
    find all session items whose Apple startTime falls within the recording's
    file-mtime window. Returns a list of ordered title_id lists.
    """
    if not os.path.isdir(RECORDINGS_DIR):
        return []

    all_items = _parse_all_items_flat(db)
    if not all_items:
        return []

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
            continue  # too short to be a real set

        rec_end   = os.path.getmtime(path)   # Unix timestamp
        rec_start = rec_end - dur

        matched = sorted(
            [(ts, tid) for ts, tid in all_items if rec_start <= ts <= rec_end],
            key=lambda x: x[0]
        )
        if len(matched) >= MIN_SESSION_TRACKS:
            track_lists.append([tid for _, tid in matched])

    return track_lists


def build_training_data(db, feature_cache, use_recordings=True):
    """Extract (feature_vector, label) pairs from djay Pro set history.

    If use_recordings=True and recordings exist, those sessions are used as the
    primary source (highest quality — every transition was intentional).
    Falls back to session history filtered to ≥ MIN_SESSION_TRACKS.

    Positives: consecutive track pairs.
    Negatives: equal-count random non-adjacent pairs from the same track lists.
    Feature vector: element-wise absolute difference of the two tracks' audio features.
    """
    # titleID → audio features (from cache, via file path)
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
        key_dist  = camelot_distance(camelot_a, camelot_b)
        return [
            bpm_diff,
            key_dist,
            abs(a['spectral_flux']      - b['spectral_flux']),
            abs(a['spectral_centroid']  - b['spectral_centroid']),
            abs(a['onset_density']      - b['onset_density']),
        ]

    # Build ordered title_id lists — prefer recordings, fall back to session history
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
        # Resolve title_ids to feature dicts, dropping tracks not in cache
        resolved = []
        for title_id in title_ids:
            feat = features_for(title_id)
            if feat:
                resolved.append(feat)

        if len(resolved) < 4:
            continue

        # Positives: every consecutive pair
        for i in range(len(resolved) - 1):
            X.append(feature_vector(resolved[i], resolved[i + 1]))
            y.append(1)

        # Negatives: random non-adjacent pairs, same count as positives
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


def train_transition_model(db, feature_cache):
    """Train a RandomForestClassifier on set history and save it to MODEL_FILE.

    Returns the trained model, or None if sklearn is unavailable or data is insufficient.
    """
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
    key_dist  = camelot_distance(a['camelot'], b['camelot'])
    vec = [[
        bpm_diff,
        key_dist,
        abs(a.get('spectral_flux', 0)      - b.get('spectral_flux', 0)),
        abs(a.get('spectral_centroid', 0)  - b.get('spectral_centroid', 0)),
        abs(a.get('onset_density', 0)      - b.get('onset_density', 0)),
    ]]
    prob_good = model.predict_proba(vec)[0][1]
    return 1.0 - prob_good   # convert probability of good transition → cost


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def camelot_distance(c1, c2):
    n1, l1 = c1
    n2, l2 = c2
    if n1 == n2 and l1 == l2:
        return 0.0
    if n1 == n2:
        return 0.5
    num_diff = min(abs(n1 - n2), 12 - abs(n1 - n2))
    return num_diff + (0.0 if l1 == l2 else 0.5)


def transition_cost(a, b, bpm_w=0.30, key_w=0.50, flux_w=0.12, spectral_w=0.08):
    bpm_diff = min(
        abs(a['tempo'] - b['tempo']),
        abs(a['tempo'] - b['tempo'] * 2),
        abs(a['tempo'] * 2 - b['tempo']),
    )
    bpm_cost      = bpm_diff / 140.0
    key_cost      = camelot_distance(a['camelot'], b['camelot']) / 6.0
    # Spectral flux gap: typical range 0–10 → normalise by 10
    flux_cost     = abs(a.get('spectral_flux', 0) - b.get('spectral_flux', 0)) / 10.0
    # Spectral centroid: typical range 500–4000 Hz → normalise by 3500
    spectral_cost = abs(a.get('spectral_centroid', 2000) - b.get('spectral_centroid', 2000)) / 3500.0
    return bpm_w * bpm_cost + key_w * key_cost + flux_w * flux_cost + spectral_w * spectral_cost


def total_cost(tracks):
    return sum(transition_cost(tracks[i], tracks[i + 1]) for i in range(len(tracks) - 1))


def _build_cost_matrix(tracks, cost_fn):
    """Precompute all pairwise costs into an n×n matrix.

    This amortises expensive cost_fn calls (e.g. ML predict_proba) so that SA
    only needs O(1) lookups per move instead of re-calling cost_fn each time.
    For n=100 this is 10,000 calls once at startup vs ~2M calls during SA.
    """
    n = len(tracks)
    # Build all (i,j) pairs as a batch for vectorised ML scoring
    pairs_i, pairs_j = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
    costs_flat = [cost_fn(tracks[i], tracks[j]) for i, j in zip(pairs_i, pairs_j)]
    matrix = [[0.0] * n for _ in range(n)]
    for (i, j), c in zip(zip(pairs_i, pairs_j), costs_flat):
        matrix[i][j] = c
    return matrix


def greedy_sort(tracks, cost_fn=None, _matrix=None):
    if not tracks:
        return []
    cost_fn   = cost_fn or transition_cost
    matrix    = _matrix if _matrix is not None else _build_cost_matrix(tracks, cost_fn)
    idx       = list(range(len(tracks)))
    by_bpm    = sorted(idx, key=lambda i: tracks[i]['tempo'])
    start     = by_bpm[len(by_bpm) // 2]
    remaining = set(idx)
    remaining.remove(start)
    ordered   = [start]
    while remaining:
        last = ordered[-1]
        best = min(remaining, key=lambda j: matrix[last][j])
        ordered.append(best)
        remaining.remove(best)
    return [tracks[i] for i in ordered]


def simulated_annealing_sort(tracks, initial_order=None, cost_fn=None,
                              T_start=1.0, T_end=0.001, alpha=0.995,
                              _matrix=None, progress_cb=None):
    """2-opt simulated annealing starting from greedy_sort (or a supplied order).

    Precomputes a full N×N cost matrix so SA iterations are pure index lookups —
    critical when cost_fn is an ML model with expensive predict_proba calls.
    Temperature schedule: geometric cooling T *= alpha each step.
    """
    cost_fn = cost_fn or transition_cost
    n       = len(tracks)
    if n < 4:
        return list(initial_order or tracks)

    # Map track objects → indices for matrix lookups
    track_to_idx = {id(t): i for i, t in enumerate(tracks)}

    if _matrix is not None:
        matrix = _matrix
    else:
        logger.info('Precomputing %d×%d cost matrix…', n, n)
        matrix = _build_cost_matrix(tracks, cost_fn)

    def _edge(a_idx, b_idx):
        return matrix[a_idx][b_idx]

    # Work in index space throughout; pass matrix to greedy to avoid rebuilding it
    if initial_order is not None:
        current = [track_to_idx[id(t)] for t in initial_order]
    else:
        current = [track_to_idx[id(t)] for t in greedy_sort(tracks, cost_fn=cost_fn, _matrix=matrix)]

    current_cost = sum(_edge(current[i], current[i + 1]) for i in range(n - 1))
    best         = current[:]
    best_cost    = current_cost

    T              = T_start
    iters_per_temp = max(n * 4, 60)
    steps          = int(math.log(T_end / T_start) / math.log(alpha))
    total_iters    = steps * iters_per_temp
    report_every   = max(total_iters // 10, 1)
    iteration      = 0

    while T > T_end:
        for _ in range(iters_per_temp):
            iteration += 1
            if iteration % report_every == 0:
                pct = iteration / total_iters * 100
                logger.debug('SA %4.0f%%  T=%.4f  best cost=%.4f', pct, T, best_cost)
                if progress_cb:
                    progress_cb(iteration / total_iters, f'SA {pct:.0f}%  cost={best_cost:.4f}')

            i, j = sorted(random.sample(range(n), 2))
            if j - i < 2:
                continue

            # O(1) delta via matrix: only the two boundary edges change.
            pre  = _edge(current[i - 1], current[i]) if i > 0 else 0
            pre += _edge(current[j], current[j + 1]) if j < n - 1 else 0
            post  = _edge(current[i - 1], current[j]) if i > 0 else 0
            post += _edge(current[i], current[j + 1]) if j < n - 1 else 0
            delta = post - pre

            if delta < 0 or random.random() < math.exp(-delta / T):
                current[i:j + 1] = current[i:j + 1][::-1]
                current_cost += delta
                if current_cost < best_cost:
                    best      = current[:]
                    best_cost = current_cost

        T *= alpha

    return [tracks[i] for i in best]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def assign_energy_levels(tracks):
    """Add an 'energy_level' field (1–5) to each track based on spectral flux quintiles.

    Ratings are relative within the playlist — the top 20% by flux get 5,
    the bottom 20% get 1. Uses rank position so duplicate flux values are
    handled correctly.
    """
    n = len(tracks)
    # argsort: indices that would sort tracks by flux
    order = sorted(range(n), key=lambda i: tracks[i].get('spectral_flux', 0))
    for rank, idx in enumerate(order):
        tracks[idx]['energy_level'] = min(5, int(rank / n * 5) + 1)
    return tracks


def print_tracklist(tracks):
    header = f"{'#':<4} {'Title':<48} {'Artist':<22} {'BPM':>6}  {'Key':<5}  {'Nrg'}  {'kHz':>6}"
    print('\n' + header)
    print('-' * len(header))
    for i, t in enumerate(tracks, 1):
        cam = f"{t['camelot'][0]}{t['camelot'][1]}"
        nrg = t.get('energy_level', '?')
        khz = t.get('spectral_centroid', 0) / 1000  # Hz → kHz
        print(
            f"{i:<4} {t['name'][:47]:<48} {t['artist'][:21]:<22} "
            f"{t['tempo']:>6.1f}  {cam:<5}  {nrg!s:>3}  {khz:>6.2f}"
        )


def export_m3u(tracks, output_path):
    """Write an Extended M3U file for the given (sorted) track list."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('#EXTM3U\n')
        for t in tracks:
            artist = t.get('artist', '')
            name   = t.get('name', os.path.basename(t['path']))
            label  = f"{artist} - {name}" if artist else name
            f.write(f'#EXTINF:-1,{label}\n')
            f.write(f"{t['path']}\n")
    print(f"M3U exported → {output_path}")


def transition_stats(tracks, label=''):
    if len(tracks) < 2:
        return
    jumps = [abs(tracks[i]['tempo'] - tracks[i + 1]['tempo'])
             for i in range(len(tracks) - 1)]
    clashes = sum(
        1 for i in range(len(tracks) - 1)
        if camelot_distance(tracks[i]['camelot'], tracks[i + 1]['camelot']) > 1
    )
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}Total cost   : {total_cost(tracks):.4f}")
    print(f"{tag}Avg BPM jump : {sum(jumps) / len(jumps):.1f}")
    print(f"{tag}Max BPM jump : {max(jumps):.1f}")
    print(f"{tag}Key clashes  : {clashes} / {len(tracks) - 1} transitions")


# ---------------------------------------------------------------------------
# TSAF binary encoding
# ---------------------------------------------------------------------------

ROOT_PLAYLIST_UUID = '60526854-5D0D-47B8-9AA6-025C7516E7F4'

def _tsaf_str(s):
    return b'\x08' + s.encode('utf-8') + b'\x00'

def encode_tsaf_playlist(pl_uuid, name):
    # djay Pro does NOT store itemUUIDs in the playlist blob — it reads tracks
    # from the relationship tables / view tables instead. Blobs with an itemUUIDs
    # array are silently rejected. Match the exact format djay Pro writes itself.
    body = (
        _tsaf_str('ADCMediaItemPlaylist')
        + _tsaf_str(pl_uuid)            + _tsaf_str('uuid')
        + _tsaf_str(name)               + _tsaf_str('name')
        + _tsaf_str(ROOT_PLAYLIST_UUID) + _tsaf_str('parentUUID')
        + b'\x2e'                       + _tsaf_str('type')
        + b'\x00'
    )
    header = b'TSAF' + struct.pack('<HH', 3, 3) + struct.pack('<Q', 1) + struct.pack('<I', 8) + b'\x2b'
    return header + body


def encode_tsaf_playlist_item(item_uuid, pl_uuid, media_key_lowercase):
    # media_key_lowercase: the key from database2 for the media item — already lowercase, no hyphens
    body = (
        _tsaf_str('ADCMediaItemPlaylistItem')
        + _tsaf_str(item_uuid)            + _tsaf_str('uuid')
        + _tsaf_str(pl_uuid)              + _tsaf_str('playlistUUID')
        + _tsaf_str(media_key_lowercase)  + _tsaf_str('mediaItemUUID')
        + b'\x00'   # single body terminator (matches djay Pro's own blobs)
    )
    header = b'TSAF' + struct.pack('<HH', 3, 3) + struct.pack('<Q', 0) + struct.pack('<I', 7) + b'\x2b'
    return header + body


# ---------------------------------------------------------------------------
# Write sorted playlist as a new clone in djay Pro's database
# ---------------------------------------------------------------------------

def _backup_djay_db():
    """Checkpoint WAL then copy .db + .db-shm + .db-wal with a timestamp."""
    # Flush WAL into main file first so the backup is self-contained
    tmp = sqlite3.connect(DJAY_DB)
    tmp.execute('PRAGMA wal_checkpoint(FULL)')
    tmp.close()

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    for suffix in ('', '-shm', '-wal'):
        src = DJAY_DB + suffix
        if os.path.exists(src):
            shutil.copy2(src, DJAY_DB + f'.{ts}.bak{suffix}')
    return DJAY_DB + f'.{ts}.bak'


def create_sorted_clone(pl_rowid, sorted_tracks, new_name):
    """
    Creates a brand-new djay Pro playlist as a sorted clone of an existing one.
    The original playlist is never touched.
    Backs up all three WAL files first, rolls back on any error.
    """

    # Safety: refuse to write if djay Pro is running (would corrupt the live DB)
    check = subprocess.run(['pgrep', '-x', 'djay Pro'], capture_output=True)
    if check.returncode == 0:
        raise RuntimeError('djay Pro is running — quit it before writing the sorted playlist.')

    backup_path = _backup_djay_db()
    print(f'Database backed up → {backup_path}')

    db = sqlite3.connect(DJAY_DB)
    try:
        # Collect the set of media rowids that belong to this playlist
        existing_media_rowids = {
            r[0] for r in db.execute('''
                SELECT r_media.dst
                FROM relationship_relationship r_item
                JOIN relationship_relationship r_media
                    ON r_media.src = r_item.src
                    AND r_media.name = "mediaItemPlaylistItemMediaItem"
                WHERE r_item.name = "mediaItemPlaylistItemPlaylist"
                  AND r_item.dst = ?
            ''', (pl_rowid,))
        }

        # Assign fresh UUIDs for the clone
        new_pl_uuid = str(uuid.uuid4()).upper()
        new_items   = [
            (str(uuid.uuid4()).upper(), t['_rowid'])
            for t in sorted_tracks
            if t['_rowid'] in existing_media_rowids
        ]
        if not new_items:
            raise ValueError('No valid tracks found in sorted list')

        pl_blob = encode_tsaf_playlist(new_pl_uuid, new_name)

        db.execute('BEGIN')

        # ── Insert new playlist ────────────────────────────────────────────
        db.execute(
            'INSERT INTO database2(collection, key, data) VALUES (?,?,?)',
            ('mediaItemPlaylists', new_pl_uuid, pl_blob),
        )
        new_pl_rowid = db.execute('SELECT last_insert_rowid()').fetchone()[0]

        db.execute(
            'INSERT INTO secondaryIndex_mediaItemPlaylistIndex(rowid, name) VALUES (?,?)',
            (new_pl_rowid, new_name),
        )

        # ── Playlist ↔ parent folder relationships ────────────────────────
        # Mixes folder rowid (parent of all Mix playlists)
        mixes_rowid = db.execute(
            'SELECT rowid FROM secondaryIndex_mediaItemPlaylistIndex WHERE name="Mixes"'
        ).fetchone()[0]
        db.execute(
            'INSERT INTO relationship_relationship(name, src, dst, rules, manual) VALUES (?,?,?,?,?)',
            ('mediaItemPlaylistParent', new_pl_rowid, mixes_rowid, 4, 0),
        )
        db.execute(
            'INSERT INTO relationship_relationship(name, src, dst, rules, manual) VALUES (?,?,?,?,?)',
            ('mediaItemPlaylistChild', mixes_rowid, new_pl_rowid, 2, 0),
        )

        # ── Insert playlist items + relationships ──────────────────────────
        new_item_rowids = []
        for new_item_uuid, media_rowid in new_items:
            media_key = db.execute(
                'SELECT key FROM database2 WHERE rowid=?', (media_rowid,)
            ).fetchone()[0]
            item_blob = encode_tsaf_playlist_item(
                new_item_uuid, new_pl_uuid,
                media_key,  # already lowercase no-hyphens in database2
            )
            db.execute(
                'INSERT INTO database2(collection, key, data) VALUES (?,?,?)',
                ('mediaItemPlaylistItems', new_item_uuid, item_blob),
            )
            ir = db.execute('SELECT last_insert_rowid()').fetchone()[0]
            new_item_rowids.append(ir)

            # Forward (playlist→item) and reverse (item→playlist) relationships
            db.execute(
                'INSERT INTO relationship_relationship(name, src, dst, rules, manual) VALUES (?,?,?,?,?)',
                ('mediaItemPlaylistItem', new_pl_rowid, ir, 2, 0),
            )
            db.execute(
                'INSERT INTO relationship_relationship(name, src, dst, rules, manual) VALUES (?,?,?,?,?)',
                ('mediaItemPlaylistItemPlaylist', ir, new_pl_rowid, 4, 0),
            )
            db.execute(
                'INSERT INTO relationship_relationship(name, src, dst, rules, manual) VALUES (?,?,?,?,?)',
                ('mediaItemPlaylistItemMediaItem', ir, media_rowid, 4, 0),
            )

        # ── Update view_mediaItemPlaylistsView (sidebar playlist list) ─────
        # New playlist is a child of ROOT_PLAYLIST_UUID — add it to that group's page.
        page_key, page_data, page_count = db.execute(
            'SELECT pageKey, data, count FROM view_mediaItemPlaylistsView_page'
            ' WHERE "group"=?', (ROOT_PLAYLIST_UUID,)
        ).fetchone()
        db.execute(
            'UPDATE view_mediaItemPlaylistsView_page SET data=?, count=? WHERE pageKey=?',
            (page_data + struct.pack('<q', new_pl_rowid), page_count + 1, page_key),
        )
        db.execute(
            'INSERT INTO view_mediaItemPlaylistsView_map(rowid, pageKey) VALUES (?,?)',
            (new_pl_rowid, page_key),
        )

        # ── Create page in view_mediaItemPlaylistView (track listing) ──────
        # group = new playlist UUID; count = number of items.
        new_view_page_key = str(uuid.uuid4()).upper()
        n = len(new_item_rowids)
        db.execute(
            'INSERT INTO view_mediaItemPlaylistView_page(pageKey, "group", prevPageKey, count, data)'
            ' VALUES (?,?,?,?,?)',
            (new_view_page_key, new_pl_uuid, None, n,
             b''.join(struct.pack('<q', r) for r in new_item_rowids)),
        )
        for ir in new_item_rowids:
            db.execute(
                'INSERT INTO view_mediaItemPlaylistView_map(rowid, pageKey) VALUES (?,?)',
                (ir, new_view_page_key),
            )

        # ── Update view_mediaView (combined library view, group='playlist') ─
        mv_page_key, mv_page_data, mv_count = db.execute(
            'SELECT pageKey, data, count FROM view_mediaView_page WHERE "group"=?',
            ('playlist',)
        ).fetchone()
        db.execute(
            'UPDATE view_mediaView_page SET data=?, count=? WHERE pageKey=?',
            (mv_page_data + struct.pack('<q', new_pl_rowid), mv_count + 1, mv_page_key),
        )
        db.execute(
            'INSERT INTO view_mediaView_map(rowid, pageKey) VALUES (?,?)',
            (new_pl_rowid, mv_page_key),
        )

        # ── FTS index entry (used for search) ─────────────────────────────
        db.execute(
            'INSERT INTO fts_searchIndex(docid, playlist) VALUES (?,?)',
            (new_pl_rowid, new_name),
        )

        # ── Integrity check before committing ─────────────────────────────
        result = db.execute('PRAGMA integrity_check').fetchone()[0]
        if result != 'ok':
            raise ValueError(f'Integrity check failed: {result}')

        db.execute('COMMIT')
        print(f"'{new_name}' created in djay Pro ({len(new_item_rowids)} tracks).")
        print('Relaunch djay Pro to see it.')

    except Exception as e:
        try:
            db.execute('ROLLBACK')
        except Exception:
            pass
        db.close()
        print(f'Error: {e} — changes rolled back, database unchanged.')
        raise

    db.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        prog='djay_sorter',
        description='Optimise a djay Pro playlist order using Camelot key + BPM + energy.',
    )
    sub = parser.add_subparsers(dest='command')

    # ── sort (default) ───────────────────────────────────────────────────────
    sort_p = sub.add_parser('sort', help='Sort a playlist and write it back to djay Pro.')
    sort_p.add_argument('playlist', nargs='?',
                        help='Playlist name or number. Omit for interactive selection.')
    sort_p.add_argument('-n', '--name',
                        help='Output playlist name (default: "<playlist> (Sorted)").')
    sort_p.add_argument('-r', '--runs', type=int, default=3,
                        help='Number of SA runs; best result is kept (default: 3).')
    sort_p.add_argument('--dry-run', action='store_true',
                        help='Show sorted tracklist without writing to djay Pro.')
    sort_p.add_argument('--export', nargs='?', const='', metavar='FILE',
                        help='Export sorted playlist as M3U (default filename: <playlist>.m3u).')

    # ── list ─────────────────────────────────────────────────────────────────
    sub.add_parser('list', help='List all playlists with track counts.')

    # ── train ────────────────────────────────────────────────────────────────
    sub.add_parser('train', help='Train the ML transition model from set history.')

    return parser, parser.parse_args()


def _resolve_playlist(playlists, choice):
    """Return (rowid, name) from a name substring or 1-based index string."""
    if choice.isdigit():
        idx = int(choice) - 1
        if not (0 <= idx < len(playlists)):
            print(f"No playlist at position {choice}.")
            sys.exit(1)
        return playlists[idx][0], playlists[idx][1]
    matches = [(r, n, c) for r, n, c in playlists if choice.lower() in n.lower()]
    if not matches:
        print(f"No playlist matching '{choice}'.")
        sys.exit(1)
    return matches[0][0], matches[0][1]


def _print_playlist_table(playlists):
    print(f"{'#':<5} {'Playlist':<40} Tracks")
    print('-' * 55)
    for i, (_, name, count) in enumerate(playlists, 1):
        print(f"{i:<5} {name:<40} {count}")


def _run_sort(db, playlist_choice, output_name, sa_runs, dry_run, export_path=None):
    playlists = list_playlists(db)
    if not playlists:
        print("No playlists with local tracks found.")
        sys.exit(1)

    if playlist_choice is None:
        _print_playlist_table(playlists)
        print("\nEnter playlist number or name: ", end='', flush=True)
        playlist_choice = input().strip()

    pl_rowid, pl_name = _resolve_playlist(playlists, playlist_choice)

    print(f"\nLoading '{pl_name}'…")
    tracks, _ = get_playlist_tracks(db, pl_rowid)
    db.close()

    if not tracks:
        print("No local tracks found in this playlist.")
        sys.exit(1)

    if len(tracks) < 2:
        print("Playlist must contain at least 2 tracks to be sorted.")
        sys.exit(1)

    missing_bpm = sum(1 for t in tracks if not t['tempo'])
    if missing_bpm:
        print(f"Note: {missing_bpm} track(s) have no BPM — they'll be sorted by key only.")

    print(f"Analysing {len(tracks)} tracks (cached results are instant)…")
    enriched = enrich_with_keys(tracks)
    if not enriched:
        print("No tracks could be analysed.")
        sys.exit(1)

    model = load_transition_model()
    if model is not None:
        cost_fn    = lambda a, b: ml_transition_cost(a, b, model)
        cost_label = 'ML'
    else:
        cost_fn    = transition_cost
        cost_label = 'heuristic'

    print(f"\nUsing {cost_label} cost function.")
    n = len(enriched)
    print(f"Precomputing {n}×{n} cost matrix…")
    matrix = _build_cost_matrix(enriched, cost_fn)

    print(f"Greedy sort ({n} tracks)…")
    greedy   = greedy_sort(enriched, cost_fn=cost_fn, _matrix=matrix)
    greedy_c = sum(cost_fn(greedy[i], greedy[i + 1]) for i in range(n - 1))
    transition_stats(greedy, label='Greedy')

    print(f"\nSimulated annealing ({sa_runs} run{'s' if sa_runs > 1 else ''}, keeping best)…")
    best_sa, best_sa_c = None, float('inf')
    for run in range(1, sa_runs + 1):
        print(f"  Run {run}/{sa_runs}")
        candidate   = simulated_annealing_sort(enriched, initial_order=greedy, cost_fn=cost_fn, _matrix=matrix)
        candidate_c = sum(cost_fn(candidate[i], candidate[i + 1]) for i in range(n - 1))
        if candidate_c < best_sa_c:
            best_sa, best_sa_c = candidate, candidate_c

    sa          = best_sa
    sa_c        = best_sa_c
    improvement = (greedy_c - sa_c) / greedy_c * 100 if greedy_c > 0 else 0
    transition_stats(sa, label='SA')
    print(f"\nImprovement  : {improvement:+.1f}%  ({greedy_c:.4f} → {sa_c:.4f})")

    assign_energy_levels(sa)
    print_tracklist(sa)

    if export_path is not None:
        if not export_path:
            safe = re.sub(r'[^\w\s-]', '', pl_name).strip().replace(' ', '_')
            export_path = f"{safe}.m3u"
        export_m3u(sa, export_path)

    if dry_run:
        print("\n--dry-run: skipping write to djay Pro.")
        return

    default_name = output_name or f"{pl_name} (Sorted)"
    if output_name is None:
        print(f"\nNew playlist name [{default_name}]: ", end='', flush=True)
        entered = input().strip()
        default_name = entered or default_name

    create_sorted_clone(pl_rowid, sa, default_name)


def main():
    parser, args = _parse_args()

    # No subcommand → interactive sort (backwards-compatible behaviour)
    if args.command is None:
        try:
            db = open_djay_db()
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
        _run_sort(db, playlist_choice=None, output_name=None, sa_runs=3, dry_run=False)
        return

    if args.command == 'list':
        try:
            db = open_djay_db()
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
        _print_playlist_table(list_playlists(db))
        db.close()
        return

    if args.command == 'train':
        try:
            db = sqlite3.connect(DJAY_DB)
        except Exception as e:
            print(e)
            sys.exit(1)
        train_transition_model(db, load_key_cache())
        db.close()
        return

    if args.command == 'sort':
        try:
            db = open_djay_db()
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
        _run_sort(
            db,
            playlist_choice=args.playlist,
            output_name=args.name,
            sa_runs=args.runs,
            dry_run=args.dry_run,
            export_path=args.export,
        )


if __name__ == '__main__':
    main()
