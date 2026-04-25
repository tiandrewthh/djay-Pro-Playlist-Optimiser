"""Constants shared across the package."""

import os

# Paths
DJAY_DB            = os.path.expanduser('~/Music/djay/djay Media Library.djayMediaLibrary/MediaLibrary.db')
CACHE_FILE         = os.path.expanduser('~/.dj_key_cache.json')
CACHE_MAX_ENTRIES  = 5000   # max tracks to keep in feature cache
MODEL_FILE         = os.path.expanduser('~/.dj_transition_model.pkl')
AUDIO_EXTS         = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.aiff', '.aif', '.mp4'}
MIN_SESSION_TRACKS = 20   # sessions shorter than this are treated as testing noise

# Audio analysis
ANALYSIS_DURATION_SECS = 120   # seconds of audio to analyse for key detection
MAX_ANALYSIS_WORKERS   = 4     # parallel threads for librosa.load

# Transition cost normalisation ranges
BPM_NORMALISATION      = 140.0   # max BPM difference for cost scaling
KEY_NORMALISATION      = 6.0     # max Camelot distance for cost scaling
FLUX_NORMALISATION     = 10.0    # typical spectral flux range
CENTROID_NORMALISATION = 3500.0  # typical spectral centroid range (Hz)

# Transition cost weights (must sum to 1.0)
BPM_W      = 0.30
KEY_W      = 0.50
FLUX_W     = 0.12
SPECTRAL_W = 0.08

# Simulated annealing defaults
SA_T_START  = 1.0
SA_T_END    = 0.001
SA_ALPHA    = 0.995

# Camelot wheel (for librosa output: Spotify-style key 0-11, mode 0/1)
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

# Krumhansl-Schmuckler key profiles
_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
          5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
          4.75, 3.98, 2.69, 3.34, 3.17]

# ML training
RECORDINGS_DIR = os.path.expanduser('~/Music/djay/Recordings')
APPLE_EPOCH_TS = 978307200  # 2001-01-01 00:00:00 as Unix timestamp
