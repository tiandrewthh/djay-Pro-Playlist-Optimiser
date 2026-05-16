"""
DJ Playlist Optimiser — djay Pro edition

Reads playlists and BPM directly from djay Pro's database.
Detects musical key using librosa (more accurate than djay Pro's key analysis).
Sorts by Camelot key compatibility + minimal BPM jumps.
Writes sorted playlist back to Apple Music (visible in djay Pro).

Usage:
    python -m djay_sorter
"""

import logging
import warnings

import librosa

# Suppress librosa's verbose deprecation/numba warnings without silencing everything
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')

# ── Database ────────────────────────────────────────────────────────────────
from .db import (
    open_djay_db,
    list_playlists,
    get_playlist_tracks,
)

# ── Audio ───────────────────────────────────────────────────────────────────
from .audio import (
    load_key_cache,
    save_key_cache,
    extract_audio_features,
    enrich_with_keys,
)

# ── ML ──────────────────────────────────────────────────────────────────────
from .ml import (
    SKLEARN_AVAILABLE,
    build_training_data,
    train_transition_model,
    load_transition_model,
    ml_transition_cost,
    _build_ml_cost_matrix,
)

# ── Sorting ─────────────────────────────────────────────────────────────────
from .sorting import (
    camelot_distance,
    transition_cost,
    total_cost,
    _build_cost_matrix,
    greedy_sort,
    simulated_annealing_sort,
    assign_energy_levels,
    sort_by_energy,
)

# ── Export ──────────────────────────────────────────────────────────────────
from .export import (
    encode_tsaf_playlist,
    encode_tsaf_playlist_item,
    export_m3u,
    create_sorted_clone,
)

# ── Constants (for backwards compatibility) ─────────────────────────────────
from .constants import (
    DJAY_DB,
    CACHE_FILE,
    CACHE_MAX_ENTRIES,
    MODEL_FILE,
    AUDIO_EXTS,
    MIN_SESSION_TRACKS,
    ANALYSIS_DURATION_SECS,
    MAX_ANALYSIS_WORKERS,
    BPM_NORMALISATION,
    KEY_NORMALISATION,
    FLUX_NORMALISATION,
    CENTROID_NORMALISATION,
    BPM_W,
    KEY_W,
    FLUX_W,
    SPECTRAL_W,
    SA_T_START,
    SA_T_END,
    SA_ALPHA,
    CAMELOT,
)

# ── TypedDicts ──────────────────────────────────────────────────────────────
from typing import TypedDict


class RawTrack(TypedDict):
    """Track as returned from get_playlist_tracks — before audio analysis."""
    name: str
    artist: str
    path: str
    tempo: float
    _rowid: int


class EnrichedTrack(RawTrack):
    """Track after enrich_with_keys — includes audio analysis results."""
    key: int
    mode: int
    camelot: tuple[int, str]
    spectral_flux: float
    spectral_centroid: float
    onset_density: float


# ── CLI entry point ─────────────────────────────────────────────────────────
from .cli import main
