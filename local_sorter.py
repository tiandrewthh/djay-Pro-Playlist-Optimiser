#!/usr/bin/env python3
"""
DJ Playlist Optimiser - Local Audio Analyser
Extracts BPM and key from a folder of audio files using librosa,
then sorts tracks by Camelot key compatibility and minimal BPM jumps.

Usage:
    python local_sorter.py /path/to/folder
"""

import sys
import os
import json
import warnings
import numpy as np

warnings.filterwarnings('ignore')  # suppress librosa/numba noise

AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
CACHE_FILE = '.analysis_cache.json'

# ---------------------------------------------------------------------------
# Camelot wheel
# ---------------------------------------------------------------------------
CAMELOT = {
    (0,  0): (5,  'A'),  # C  minor
    (1,  0): (12, 'A'),  # C# minor
    (2,  0): (7,  'A'),  # D  minor
    (3,  0): (2,  'A'),  # Eb minor
    (4,  0): (9,  'A'),  # E  minor
    (5,  0): (4,  'A'),  # F  minor
    (6,  0): (11, 'A'),  # F# minor
    (7,  0): (6,  'A'),  # G  minor
    (8,  0): (1,  'A'),  # Ab minor
    (9,  0): (8,  'A'),  # A  minor
    (10, 0): (3,  'A'),  # Bb minor
    (11, 0): (10, 'A'),  # B  minor
    (0,  1): (8,  'B'),  # C  major
    (1,  1): (3,  'B'),  # Db major
    (2,  1): (10, 'B'),  # D  major
    (3,  1): (5,  'B'),  # Eb major
    (4,  1): (12, 'B'),  # E  major
    (5,  1): (7,  'B'),  # F  major
    (6,  1): (2,  'B'),  # F# major
    (7,  1): (9,  'B'),  # G  major
    (8,  1): (4,  'B'),  # Ab major
    (9,  1): (11, 'B'),  # A  major
    (10, 1): (6,  'B'),  # Bb major
    (11, 1): (1,  'B'),  # B  major
}

# Krumhansl-Schmuckler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------

def detect_bpm_and_key(path):
    import librosa
    y, sr = librosa.load(path, mono=True, duration=120)  # first 2 min is enough

    # BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    # Key via Krumhansl-Schmuckler
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)

    best_score, best_key, best_mode = -np.inf, 0, 1
    for k in range(12):
        for mode, profile in ((1, _MAJOR), (0, _MINOR)):
            score = float(np.corrcoef(mean_chroma, np.roll(profile, k))[0, 1])
            if score > best_score:
                best_score, best_key, best_mode = score, k, mode

    return bpm, best_key, best_mode


def analyse_folder(folder):
    cache_path = os.path.join(folder, CACHE_FILE)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    files = sorted(
        p for p in os.listdir(folder)
        if os.path.splitext(p)[1].lower() in AUDIO_EXTS
        and not p.endswith('.crdownload.wav')   # skip partial downloads
    )

    tracks = []
    changed = False

    for i, filename in enumerate(files):
        path = os.path.join(folder, filename)
        mtime = str(os.path.getmtime(path))

        if filename in cache and cache[filename].get('mtime') == mtime:
            data = cache[filename]
        else:
            print(f"  [{i+1}/{len(files)}] Analysing: {filename}")
            try:
                bpm, key, mode = detect_bpm_and_key(path)
            except Exception as e:
                print(f"    Skipped ({e})")
                continue
            data = {'bpm': bpm, 'key': key, 'mode': mode, 'mtime': mtime}
            cache[filename] = data
            changed = True

        camelot = CAMELOT.get((data['key'], data['mode']), (0, '?'))
        tracks.append({
            'name':    os.path.splitext(filename)[0],
            'file':    filename,
            'tempo':   data['bpm'],
            'key':     data['key'],
            'mode':    data['mode'],
            'camelot': camelot,
        })

    if changed:
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)

    return tracks


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


def transition_cost(a, b, bpm_w=0.35, key_w=0.65):
    bpm_diff = min(
        abs(a['tempo'] - b['tempo']),
        abs(a['tempo'] - b['tempo'] * 2),
        abs(a['tempo'] * 2 - b['tempo']),
    )
    bpm_cost = bpm_diff / 140.0
    key_cost = camelot_distance(a['camelot'], b['camelot']) / 6.0
    return bpm_w * bpm_cost + key_w * key_cost


def greedy_sort(tracks):
    if not tracks:
        return []
    by_bpm = sorted(tracks, key=lambda t: t['tempo'])
    start = by_bpm[len(by_bpm) // 2]
    remaining = list(tracks)
    remaining.remove(start)
    ordered = [start]
    while remaining:
        best = min(remaining, key=lambda t: transition_cost(ordered[-1], t))
        ordered.append(best)
        remaining.remove(best)
    return ordered


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_tracklist(tracks):
    header = f"{'#':<4} {'Title':<50} {'BPM':>6}  {'Key':<5}"
    print('\n' + header)
    print('-' * len(header))
    for i, t in enumerate(tracks, 1):
        cam = f"{t['camelot'][0]}{t['camelot'][1]}"
        print(f"{i:<4} {t['name'][:49]:<50} {t['tempo']:>6.1f}  {cam:<5}")


def transition_stats(tracks):
    if len(tracks) < 2:
        return
    jumps = [abs(tracks[i]['tempo'] - tracks[i+1]['tempo']) for i in range(len(tracks)-1)]
    clashes = sum(
        1 for i in range(len(tracks)-1)
        if camelot_distance(tracks[i]['camelot'], tracks[i+1]['camelot']) > 1
    )
    print(f"\nAvg BPM jump : {sum(jumps)/len(jumps):.1f}")
    print(f"Max BPM jump : {max(jumps):.1f}")
    print(f"Key clashes  : {clashes} / {len(tracks)-1} transitions")


def write_m3u(tracks, folder, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('#EXTM3U\n')
        for t in tracks:
            cam = f"{t['camelot'][0]}{t['camelot'][1]}"
            f.write(f"#EXTINF:-1,{t['name']}  [{t['tempo']:.0f} BPM | {cam}]\n")
            f.write(os.path.join(folder, t['file']) + '\n')
    print(f"Playlist saved → {out_path}")


def create_apple_music_playlist(tracks, folder, playlist_name):
    import tempfile, subprocess

    # Build the file references — escape single quotes in paths
    def esc(p):
        return p.replace('\\', '\\\\').replace('"', '\\"')

    file_lines = '\n'.join(
        f'        add (POSIX file "{esc(os.path.join(folder, t["file"]))}") to newPL'
        for t in tracks
    )

    script = f'''
tell application "Music"
    try
        delete (first user playlist whose name is "{playlist_name}")
    end try
    set newPL to (make new user playlist with properties {{name:"{playlist_name}"}})
{file_lines}
end tell
'''

    with tempfile.NamedTemporaryFile(suffix='.applescript', mode='w', delete=False) as f:
        f.write(script)
        tmp = f.name

    result = subprocess.run(['osascript', tmp], capture_output=True, text=True)
    os.unlink(tmp)

    if result.returncode != 0:
        print(f"Apple Music error: {result.stderr.strip()}")
    else:
        print(f"Playlist '{playlist_name}' created in Apple Music — open djay Pro to find it.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python local_sorter.py /path/to/audio/folder")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Not a directory: {folder}")
        sys.exit(1)

    print(f"Scanning {folder}…")
    tracks = analyse_folder(folder)

    if not tracks:
        print("No audio files found.")
        sys.exit(1)

    print(f"Sorting {len(tracks)} tracks…")
    sorted_tracks = greedy_sort(tracks)

    print_tracklist(sorted_tracks)
    transition_stats(sorted_tracks)

    folder_name = os.path.basename(folder.rstrip('/'))
    m3u_path = os.path.join(folder, f"{folder_name}_sorted.m3u")
    write_m3u(sorted_tracks, folder, m3u_path)
    create_apple_music_playlist(sorted_tracks, folder, f"{folder_name} (Sorted)")


if __name__ == '__main__':
    main()
