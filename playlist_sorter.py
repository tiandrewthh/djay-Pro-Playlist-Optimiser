#!/usr/bin/env python3
"""
DJ Playlist Optimiser - Phase 1
Greedy sort by Camelot key compatibility and minimal BPM jumps.

Usage:
    python playlist_sorter.py <spotify_playlist_url_or_id>
"""

import sys
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Load credentials from .env/.env.example (or .env if it exists)
load_dotenv(dotenv_path='.env/.env.example')

# ---------------------------------------------------------------------------
# Camelot wheel
# Spotify key: 0=C 1=C# 2=D 3=D# 4=E 5=F 6=F# 7=G 8=G# 9=A 10=A# 11=B
# Spotify mode: 0=minor  1=major
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


def camelot_distance(c1, c2):
    """
    Harmonic distance between two Camelot positions.
    0   = same key
    0.5 = relative major/minor (same number, A<->B)
    1   = one step around the wheel (compatible)
    2+  = increasingly incompatible
    """
    n1, l1 = c1
    n2, l2 = c2
    if n1 == n2 and l1 == l2:
        return 0.0
    if n1 == n2:          # relative major/minor
        return 0.5
    num_diff = min(abs(n1 - n2), 12 - abs(n1 - n2))
    letter_penalty = 0.0 if l1 == l2 else 0.5
    return num_diff + letter_penalty


def transition_cost(a, b, bpm_w=0.35, key_w=0.65):
    """
    Weighted cost of transitioning from track a -> b.  Lower = better.
    BPM: also accepts half/double-time compatible tempos.
    """
    bpm_diff = min(
        abs(a['tempo'] - b['tempo']),
        abs(a['tempo'] - b['tempo'] * 2),
        abs(a['tempo'] * 2 - b['tempo']),
    )
    bpm_cost = bpm_diff / 140.0          # normalise over typical range

    key_cost = camelot_distance(a['camelot'], b['camelot']) / 6.0  # normalise 0–1

    return bpm_w * bpm_cost + key_w * key_cost


def greedy_sort(tracks):
    """
    Nearest-neighbour greedy sort.
    Starts from the track closest to median energy (a natural set midpoint).
    """
    if not tracks:
        return []

    by_energy = sorted(tracks, key=lambda t: t['energy'])
    start = by_energy[len(by_energy) // 2]

    remaining = list(tracks)
    remaining.remove(start)
    ordered = [start]

    while remaining:
        current = ordered[-1]
        best = min(remaining, key=lambda t: transition_cost(current, t))
        ordered.append(best)
        remaining.remove(best)

    return ordered


# ---------------------------------------------------------------------------
# Spotify helpers
# ---------------------------------------------------------------------------

def playlist_id_from_input(raw):
    """Accept a full URL or bare playlist ID."""
    if 'spotify.com' in raw:
        return raw.split('playlist/')[-1].split('?')[0]
    return raw


def fetch_all_tracks(sp, playlist_id):
    tracks, page = [], sp.playlist_tracks(playlist_id)
    while page:
        for item in page['items']:
            # Spotify API returns the track under 'track' or 'item' depending on version
            t = item.get('track') or item.get('item')
            if t and t.get('id') and t.get('type') == 'track':
                tracks.append(t)
        page = sp.next(page) if page['next'] else None
    return tracks


def fetch_audio_features(sp, tracks):
    ids = [t['id'] for t in tracks]
    feats = []
    for i in range(0, len(ids), 100):
        feats.extend(sp.audio_features(ids[i:i + 100]))
    return feats


def build_track_data(tracks, features):
    result = []
    for track, feat in zip(tracks, features):
        if not feat:
            continue
        key, mode = feat.get('key', -1), feat.get('mode', -1)
        camelot = CAMELOT.get((key, mode), (0, '?'))
        result.append({
            'id':           track['id'],
            'name':         track['name'],
            'artist':       track['artists'][0]['name'],
            'tempo':        feat.get('tempo', 0.0),
            'key':          key,
            'mode':         mode,
            'camelot':      camelot,
            'energy':       feat.get('energy', 0.0),
            'danceability': feat.get('danceability', 0.0),
            'valence':      feat.get('valence', 0.0),
        })
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_tracklist(tracks):
    header = f"{'#':<4} {'Title':<44} {'Artist':<24} {'BPM':>6}  {'Key':<5} {'Energy':>7} {'Dance':>6} {'Valence':>8}"
    print('\n' + header)
    print('-' * len(header))
    for i, t in enumerate(tracks, 1):
        cam = f"{t['camelot'][0]}{t['camelot'][1]}"
        print(
            f"{i:<4} {t['name'][:43]:<44} {t['artist'][:23]:<24} "
            f"{t['tempo']:>6.1f}  {cam:<5} {t['energy']:>7.2f} "
            f"{t['danceability']:>6.2f} {t['valence']:>8.2f}"
        )


def transition_stats(tracks):
    if len(tracks) < 2:
        return
    jumps = [abs(tracks[i]['tempo'] - tracks[i+1]['tempo']) for i in range(len(tracks)-1)]
    avg = sum(jumps) / len(jumps)
    worst = max(jumps)
    incompatible = sum(
        1 for i in range(len(tracks)-1)
        if camelot_distance(tracks[i]['camelot'], tracks[i+1]['camelot']) > 1
    )
    print(f"\nTransition quality:")
    print(f"  Avg BPM jump : {avg:.1f}")
    print(f"  Max BPM jump : {worst:.1f}")
    print(f"  Key clashes  : {incompatible} / {len(tracks)-1} transitions")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python playlist_sorter.py <playlist_url_or_id>")
        sys.exit(1)

    playlist_id = playlist_id_from_input(sys.argv[1])

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        scope='playlist-read-private playlist-modify-public playlist-modify-private',
        redirect_uri=os.getenv('SPOTIPY_REDIRECT_URI', 'http://127.0.0.1:8888/callback'),
    ))

    playlist_meta = sp.playlist(playlist_id)
    print(f"Playlist : {playlist_meta['name']}")
    total = (playlist_meta.get('tracks') or {}).get('total', '?')
    print(f"Tracks   : {total}")

    print("Fetching tracks…")
    tracks = fetch_all_tracks(sp, playlist_id)

    print("Fetching audio features…")
    features = fetch_audio_features(sp, tracks)
    track_data = build_track_data(tracks, features)

    print(f"Sorting {len(track_data)} tracks…")
    sorted_tracks = greedy_sort(track_data)

    print_tracklist(sorted_tracks)
    transition_stats(sorted_tracks)

    original_count = len(tracks)
    sorted_count = len(sorted_tracks)
    if sorted_count == 0:
        print("\nNo tracks had audio features — aborting without touching the playlist.")
        sys.exit(1)
    if sorted_count < original_count:
        skipped = original_count - sorted_count
        print(f"\nWarning: {skipped} track(s) were skipped (no audio features — likely local files).")

    answer = input(f"\nReorder {sorted_count} tracks on Spotify? [y/N] ").strip().lower()
    if answer == 'y':
        sorted_ids = [t['id'] for t in sorted_tracks]
        sp.playlist_replace_items(playlist_id, sorted_ids[:100])
        for i in range(100, len(sorted_ids), 100):
            sp.playlist_add_items(playlist_id, sorted_ids[i:i + 100])
        print("Done — playlist reordered on Spotify.")


if __name__ == '__main__':
    main()
