"""Audio feature extraction — key detection, cache management, enrichment."""

import json
import logging
import os
import threading

import librosa
import numpy as np

from .constants import (
    ANALYSIS_DURATION_SECS,
    CACHE_FILE,
    CACHE_MAX_ENTRIES,
    CAMELOT,
    MAX_ANALYSIS_WORKERS,
    _MAJOR,
    _MINOR,
)

logger = logging.getLogger(__name__)


def load_key_cache():
    """Load the cached audio feature cache from disk.

    Returns:
        Dict mapping file paths to cached feature dicts. Returns empty dict
        if the cache file does not exist.
    """
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_key_cache(cache):
    """Persist the audio feature cache to disk, evicting old entries if full.

    If the cache exceeds CACHE_MAX_ENTRIES entries, the oldest entries
    (by modification time) are removed before writing.

    Args:
        cache: Dict mapping file paths to cached feature dicts.
    """
    if len(cache) > CACHE_MAX_ENTRIES:
        sorted_paths = sorted(cache.keys(), key=lambda p: cache.get(p, {}).get('mtime', '0'))
        to_remove = len(cache) - CACHE_MAX_ENTRIES
        for path in sorted_paths[:to_remove]:
            del cache[path]
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def extract_audio_features(path):
    """Extract key, mode, spectral flux, spectral centroid, onset density, and BPM."""
    y, sr = librosa.load(path, mono=True, duration=ANALYSIS_DURATION_SECS)

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
    """Add audio features to tracks, using cache where available.

    Returns list of EnrichedTrack dicts with key, mode, camelot, spectral_flux,
    spectral_centroid, onset_density fields.
    """
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
    workers = min(MAX_ANALYSIS_WORKERS, len(to_analyse)) if to_analyse else 1
    import concurrent.futures
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
