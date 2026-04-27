"""CLI — argument parsing, interactive sort, main entry point."""

import logging
import re
import sys

from .constants import DJAY_DB
from .db import open_djay_db, list_playlists, get_playlist_tracks
from .audio import enrich_with_keys, load_key_cache
from .ml import load_transition_model, ml_transition_cost, train_transition_model
from .sorting import (
    transition_cost,
    total_cost,
    _build_cost_matrix,
    greedy_sort,
    simulated_annealing_sort,
    assign_energy_levels,
)
from .export import export_m3u, create_sorted_clone

logger = logging.getLogger(__name__)


def _parse_args():
    import argparse
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


def _print_tracklist(tracks):
    header = f"{'#':<4} {'Title':<48} {'Artist':<22} {'BPM':>6}  {'Key':<5}  {'Nrg'}  {'kHz':>6}"
    print('\n' + header)
    print('-' * len(header))
    for i, t in enumerate(tracks, 1):
        cam = f"{t['camelot'][0]}{t['camelot'][1]}"
        nrg = t.get('energy_level', '?')
        khz = t.get('spectral_centroid', 0) / 1000
        print(
            f"{i:<4} {t['name'][:47]:<48} {t['artist'][:21]:<22} "
            f"{t['tempo']:>6.1f}  {cam:<5}  {nrg!s:>3}  {khz:>6.2f}"
        )


def _transition_stats(tracks, label=''):
    if len(tracks) < 2:
        return
    jumps = [abs(tracks[i]['tempo'] - tracks[i + 1]['tempo'])
             for i in range(len(tracks) - 1)]
    clashes = sum(
        1 for i in range(len(tracks) - 1)
        if _camelot_distance_for_stats(tracks[i]['camelot'], tracks[i + 1]['camelot']) > 1
    )
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}Total cost   : {total_cost(tracks):.4f}")
    print(f"{tag}Avg BPM jump : {sum(jumps) / len(jumps):.1f}")
    print(f"{tag}Max BPM jump : {max(jumps):.1f}")
    print(f"{tag}Key clashes  : {clashes} / {len(tracks) - 1} transitions")


def _camelot_distance_for_stats(c1, c2):
    """Inline camelot distance for CLI stats (avoids circular import)."""
    n1, l1 = c1
    n2, l2 = c2
    if n1 == n2 and l1 == l2:
        return 0.0
    if n1 == n2:
        return 0.5
    num_diff = min(abs(n1 - n2), 12 - abs(n1 - n2))
    return num_diff + (0.0 if l1 == l2 else 0.5)


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
    _transition_stats(greedy, label='Greedy')

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
    _transition_stats(sa, label='SA')
    print(f"\nImprovement  : {improvement:+.1f}%  ({greedy_c:.4f} → {sa_c:.4f})")

    assign_energy_levels(sa)
    _print_tracklist(sa)

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
    """CLI entry point — dispatches to sort, list, or train subcommands.
    
    If no subcommand is given, falls back to interactive sort mode where
    the user is prompted for playlist selection and output name.
    
    Subcommands:
        sort <playlist>  — Sort a playlist by name or ID
        list             — Show all playlists in the library
        train            — Train an ML transition model from set history
    """
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
        import sqlite3
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
