# DJ Playlist Optimiser

Reorders djay Pro playlists for smoother DJ sets using Camelot key compatibility, BPM progression, and energy flow. Runs a greedy sort followed by simulated annealing to find the best track order, then optionally writes the result back to djay Pro.

## Requirements

- macOS (djay Pro stores its library in `~/Music/djay/`)
- [djay Pro](https://www.algoriddim.com/djay-pro-mac) with at least one playlist containing local audio files
- Python 3.9+ **or** Docker

Audio analysis requires `ffmpeg` for MP3/M4A/AAC files. On macOS: `brew install ffmpeg`.

## Quick start

```bash
git clone https://github.com/tiandrewthh/DJ-Playlist-Optimiser
cd "DJ Playlist Optimiser"

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn api:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

## Configuration

| Setting | How to set |
|---|---|
| djay Pro library path | Auto-detected. Override via the ⚙ button in the UI, or set a custom path in the UI's Library Configuration panel. |
| CORS origins | `CORS_ORIGINS=https://example.com uvicorn api:app` (comma-separated; defaults to localhost) |

## How it works

1. **Read** — queries the djay Pro SQLite database for playlist tracks and BPM data
2. **Analyse** — extracts musical key (Krumhansl-Schmuckler profiles via librosa), spectral flux, and onset density from each audio file; results are cached in `~/.dj_key_cache.json`
3. **Sort** — builds a pairwise transition cost matrix, runs a greedy nearest-neighbour sort, then refines with simulated annealing
4. **Output** — displays the sorted tracklist with BPM and key transition annotations; optionally exports as M3U or writes back to djay Pro

If you have djay Pro set history, the app can train a `RandomForestClassifier` on your actual mixing decisions to replace the heuristic cost function with a personalised ML model.

## ML model training

From the command line:

```bash
source venv/bin/activate
python3 djay_sorter.py train
```

Requires at least 200 historical transitions in djay Pro's session history (or recorded sets in `~/Music/djay/Recordings/`). The model is saved to `~/.dj_transition_model.pkl` and picked up automatically on the next sort.
