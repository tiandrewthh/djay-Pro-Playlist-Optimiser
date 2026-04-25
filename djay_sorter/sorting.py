"""Sorting algorithms — cost functions, greedy nearest-neighbour, simulated annealing."""

import logging
import math
import random

from .constants import (
    BPM_NORMALISATION,
    BPM_W,
    CENTROID_NORMALISATION,
    FLUX_NORMALISATION,
    FLUX_W,
    KEY_NORMALISATION,
    KEY_W,
    SA_ALPHA,
    SA_T_END,
    SA_T_START,
    SPECTRAL_W,
)

logger = logging.getLogger(__name__)


def camelot_distance(c1, c2):
    """Camelot wheel distance between two (number, letter) key tuples."""
    n1, l1 = c1
    n2, l2 = c2
    if n1 == n2 and l1 == l2:
        return 0.0
    if n1 == n2:
        return 0.5
    num_diff = min(abs(n1 - n2), 12 - abs(n1 - n2))
    return num_diff + (0.0 if l1 == l2 else 0.5)


def transition_cost(a, b, bpm_w=BPM_W, key_w=KEY_W, flux_w=FLUX_W, spectral_w=SPECTRAL_W):
    """Heuristic transition cost between two enriched tracks."""
    bpm_diff = min(
        abs(a['tempo'] - b['tempo']),
        abs(a['tempo'] - b['tempo'] * 2),
        abs(a['tempo'] * 2 - b['tempo']),
    )
    bpm_cost      = bpm_diff / BPM_NORMALISATION
    key_cost      = camelot_distance(a['camelot'], b['camelot']) / KEY_NORMALISATION
    flux_cost     = abs(a.get('spectral_flux', 0) - b.get('spectral_flux', 0)) / FLUX_NORMALISATION
    spectral_cost = abs(a.get('spectral_centroid', 2000) - b.get('spectral_centroid', 2000)) / CENTROID_NORMALISATION
    return bpm_w * bpm_cost + key_w * key_cost + flux_w * flux_cost + spectral_w * spectral_cost


def total_cost(tracks):
    """Sum of all pairwise transition costs in a track sequence."""
    return sum(transition_cost(tracks[i], tracks[i + 1]) for i in range(len(tracks) - 1))


def _build_cost_matrix(tracks, cost_fn):
    """Precompute all pairwise costs into an n×n matrix."""
    n = len(tracks)
    matrix = [[0.0] * n for _ in range(n)]
    if n < 2:
        return matrix
    pairs_i, pairs_j = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
    costs_flat = [cost_fn(tracks[i], tracks[j]) for i, j in zip(pairs_i, pairs_j)]
    for (i, j), c in zip(zip(pairs_i, pairs_j), costs_flat):
        matrix[i][j] = c
    return matrix


def greedy_sort(tracks, cost_fn=None, _matrix=None):
    """Greedy nearest-neighbour sort starting from median-BPM track."""
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
                              T_start=SA_T_START, T_end=SA_T_END, alpha=SA_ALPHA,
                              _matrix=None, progress_cb=None):
    """2-opt simulated annealing starting from greedy_sort (or a supplied order)."""
    cost_fn = cost_fn or transition_cost
    n       = len(tracks)
    if n < 4:
        return list(initial_order or tracks)

    track_to_idx = {id(t): i for i, t in enumerate(tracks)}

    if _matrix is not None:
        matrix = _matrix
    else:
        logger.info('Precomputing %d×%d cost matrix…', n, n)
        matrix = _build_cost_matrix(tracks, cost_fn)

    def _edge(a_idx, b_idx):
        return matrix[a_idx][b_idx]

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


def assign_energy_levels(tracks):
    """Add an 'energy_level' field (1–5) to each track based on spectral flux quintiles."""
    n = len(tracks)
    order = sorted(range(n), key=lambda i: tracks[i].get('spectral_flux', 0))
    for rank, idx in enumerate(order):
        tracks[idx]['energy_level'] = min(5, int(rank / n * 5) + 1)
    return tracks
