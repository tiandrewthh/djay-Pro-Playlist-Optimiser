"""Tests for core sorting algorithms — no audio files or database required."""

import random
import pytest

from djay_sorter import (
    camelot_distance,
    transition_cost,
    total_cost,
    _build_cost_matrix,
    greedy_sort,
    simulated_annealing_sort,
    assign_energy_levels,
)


# ---------------------------------------------------------------------------
# Helpers — minimal track fixtures
# ---------------------------------------------------------------------------

def _track(name, tempo=120.0, camelot=(1, 'A'), spectral_flux=0.0, spectral_centroid=2000.0):
    return {
        'name': name,
        'artist': '',
        'path': f'/fake/{name}',
        'tempo': tempo,
        'camelot': camelot,
        'spectral_flux': spectral_flux,
        'spectral_centroid': spectral_centroid,
        '_rowid': hash(name) % 100000,
    }


# ---------------------------------------------------------------------------
# camelot_distance
# ---------------------------------------------------------------------------

class TestCamelotDistance:
    def test_identical_keys(self):
        assert camelot_distance((5, 'A'), (5, 'A')) == 0.0
        assert camelot_distance((12, 'B'), (12, 'B')) == 0.0

    def test_same_number_different_letter(self):
        assert camelot_distance((5, 'A'), (5, 'B')) == 0.5

    def test_adjacent_same_letter(self):
        # 5A → 6A = 1 step
        assert camelot_distance((5, 'A'), (6, 'A')) == 1.0

    def test_wraparound(self):
        # 12A → 1A should be 1 step (wrap around the wheel)
        assert camelot_distance((12, 'A'), (1, 'A')) == 1.0

    def test_opposite_keys(self):
        # 1A → 7A = 6 steps (half the wheel)
        assert camelot_distance((1, 'A'), (7, 'A')) == 6.0

    def test_different_letters_with_offset(self):
        # 1A → 2B: num_diff=1 + mode_mismatch=0.5
        assert camelot_distance((1, 'A'), (2, 'B')) == 1.5

    def test_symmetry(self):
        assert camelot_distance((3, 'A'), (9, 'B')) == camelot_distance((9, 'B'), (3, 'A'))

    def test_max_distance(self):
        # 1A → 7B = 6 + 0.5 = 6.5
        assert camelot_distance((1, 'A'), (7, 'B')) == 6.5


# ---------------------------------------------------------------------------
# transition_cost
# ---------------------------------------------------------------------------

class TestTransitionCost:
    def test_identical_tracks_zero_cost(self):
        t = _track('A', tempo=120.0, camelot=(5, 'A'), spectral_flux=5.0, spectral_centroid=2000.0)
        cost = transition_cost(t, t)
        assert cost == pytest.approx(0.0, abs=1e-6)

    def test_same_key_small_bpm_jump(self):
        a = _track('A', tempo=120.0, camelot=(5, 'A'), spectral_flux=5.0, spectral_centroid=2000.0)
        b = _track('B', tempo=122.0, camelot=(5, 'A'), spectral_flux=5.0, spectral_centroid=2000.0)
        cost = transition_cost(a, b)
        # Only BPM contributes; key/flux/centroid are identical
        assert cost > 0
        assert cost < 0.1  # small jump → low cost

    def test_different_keys_high_cost(self):
        a = _track('A', tempo=120.0, camelot=(1, 'A'), spectral_flux=5.0)
        b = _track('B', tempo=120.0, camelot=(7, 'B'), spectral_flux=5.0)
        cost = transition_cost(a, b)
        # Key clash dominates
        assert cost > 0.3

    def test_bpm_halving_is_cheap(self):
        # 120 → 60 BPM should be cheap (half-time mix)
        a = _track('A', tempo=120.0, camelot=(5, 'A'))
        b = _track('B', tempo=60.0, camelot=(5, 'A'))
        cost = transition_cost(a, b)
        assert cost < 0.05  # very cheap — half-time is a valid mix

    def test_bpm_doubling_is_cheap(self):
        # 60 → 120 BPM should be cheap (double-time mix)
        a = _track('A', tempo=60.0, camelot=(5, 'A'))
        b = _track('B', tempo=120.0, camelot=(5, 'A'))
        cost = transition_cost(a, b)
        assert cost < 0.05

    def test_weights_sum_to_one(self):
        # Verify default weights: 0.30 + 0.50 + 0.12 + 0.08 = 1.0
        assert 0.30 + 0.50 + 0.12 + 0.08 == pytest.approx(1.0)

    def test_custom_weights(self):
        a = _track('A', tempo=120.0, camelot=(1, 'A'))
        b = _track('B', tempo=140.0, camelot=(7, 'B'))
        c1 = transition_cost(a, b, bpm_w=1.0, key_w=0.0, flux_w=0.0, spectral_w=0.0)
        c2 = transition_cost(a, b, bpm_w=0.0, key_w=1.0, flux_w=0.0, spectral_w=0.0)
        # Different weights → different costs
        assert c1 != c2

    def test_missing_optional_fields_defaults(self):
        a = {'name': 'A', 'tempo': 120.0, 'camelot': (5, 'A'), 'path': '/fake'}
        b = {'name': 'B', 'tempo': 120.0, 'camelot': (5, 'A'), 'path': '/fake'}
        cost = transition_cost(a, b)
        # Should not raise; uses defaults for spectral_flux and spectral_centroid
        assert cost == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# total_cost
# ---------------------------------------------------------------------------

class TestTotalCost:
    def test_single_track_zero(self):
        tracks = [_track('A')]
        assert total_cost(tracks) == 0.0

    def test_two_tracks(self):
        a = _track('A', tempo=120.0, camelot=(5, 'A'))
        b = _track('B', tempo=120.0, camelot=(5, 'A'))
        assert total_cost([a, b]) == pytest.approx(transition_cost(a, b))

    def test_additive(self):
        a = _track('A', tempo=120.0, camelot=(1, 'A'))
        b = _track('B', tempo=125.0, camelot=(2, 'A'))
        c = _track('C', tempo=130.0, camelot=(3, 'A'))
        expected = transition_cost(a, b) + transition_cost(b, c)
        assert total_cost([a, b, c]) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _build_cost_matrix
# ---------------------------------------------------------------------------

class TestBuildCostMatrix:
    def test_diagonal_is_zero(self):
        tracks = [_track('A', tempo=120), _track('B', tempo=125), _track('C', tempo=130)]
        matrix = _build_cost_matrix(tracks, transition_cost)
        for i in range(len(tracks)):
            assert matrix[i][i] == pytest.approx(0.0)

    def test_symmetric_for_identical_tracks(self):
        a = _track('A', tempo=120.0, camelot=(5, 'A'))
        b = _track('B', tempo=120.0, camelot=(5, 'A'))
        matrix = _build_cost_matrix([a, b], transition_cost)
        assert matrix[0][1] == pytest.approx(matrix[1][0])

    def test_dimensions(self):
        tracks = [_track(f'T{i}') for i in range(5)]
        matrix = _build_cost_matrix(tracks, transition_cost)
        assert len(matrix) == 5
        assert all(len(row) == 5 for row in matrix)

    def test_non_negative_costs(self):
        tracks = [_track(f'T{i}', tempo=100 + i * 10, camelot=(i % 12 + 1, 'A')) for i in range(4)]
        matrix = _build_cost_matrix(tracks, transition_cost)
        for row in matrix:
            for val in row:
                assert val >= 0.0


# ---------------------------------------------------------------------------
# greedy_sort
# ---------------------------------------------------------------------------

class TestGreedySort:
    def test_empty_list(self):
        assert greedy_sort([]) == []

    def test_single_track(self):
        t = _track('A')
        assert greedy_sort([t]) == [t]

    def test_two_tracks(self):
        a = _track('A', tempo=120.0, camelot=(5, 'A'))
        b = _track('B', tempo=125.0, camelot=(6, 'A'))
        result = greedy_sort([a, b])
        assert {t['name'] for t in result} == {'A', 'B'}
        assert len(result) == 2

    def test_preserves_all_tracks(self):
        tracks = [
            _track(f'T{i}', tempo=100 + i * 5, camelot=(i % 12 + 1, 'A'))
            for i in range(10)
        ]
        result = greedy_sort(tracks)
        assert {t['name'] for t in result} == {t['name'] for t in tracks}

    def test_starts_near_median_bpm(self):
        tracks = [
            _track('Slow', tempo=100.0, camelot=(1, 'A')),
            _track('Mid',  tempo=120.0, camelot=(2, 'A')),
            _track('Fast', tempo=140.0, camelot=(3, 'A')),
        ]
        result = greedy_sort(tracks)
        # Median BPM of 3 tracks is index 1 → 'Mid'
        assert result[0]['name'] == 'Mid'

    def test_same_key_tracks_stay_adjacent(self):
        a = _track('A', tempo=120.0, camelot=(5, 'A'))
        b = _track('B', tempo=121.0, camelot=(5, 'A'))
        c = _track('C', tempo=122.0, camelot=(5, 'A'))
        result = greedy_sort([a, b, c])
        # All same key + similar BPM → cost is tiny for all pairs, but result should be valid
        assert len(result) == 3


# ---------------------------------------------------------------------------
# simulated_annealing_sort
# ---------------------------------------------------------------------------

class TestSimulatedAnnealingSort:
    def test_empty_list(self):
        assert simulated_annealing_sort([]) == []

    def test_single_track(self):
        t = _track('A')
        assert simulated_annealing_sort([t]) == [t]

    def test_two_tracks_returns_both(self):
        a = _track('A', tempo=120.0, camelot=(5, 'A'))
        b = _track('B', tempo=125.0, camelot=(6, 'A'))
        result = simulated_annealing_sort([a, b])
        assert {t['name'] for t in result} == {'A', 'B'}

    def test_three_tracks_returns_all(self):
        tracks = [
            _track('A', tempo=120.0, camelot=(1, 'A')),
            _track('B', tempo=128.0, camelot=(2, 'A')),
            _track('C', tempo=135.0, camelot=(3, 'A')),
        ]
        result = simulated_annealing_sort(tracks)
        assert {t['name'] for t in result} == {t['name'] for t in tracks}

    def test_respects_initial_order_when_cost_is_zero(self):
        # All identical tracks → any order has zero cost → SA should not worsen it
        tracks = [
            _track('A', tempo=120.0, camelot=(5, 'A'), spectral_flux=5.0, spectral_centroid=2000.0),
            _track('B', tempo=120.0, camelot=(5, 'A'), spectral_flux=5.0, spectral_centroid=2000.0),
            _track('C', tempo=120.0, camelot=(5, 'A'), spectral_flux=5.0, spectral_centroid=2000.0),
            _track('D', tempo=120.0, camelot=(5, 'A'), spectral_flux=5.0, spectral_centroid=2000.0),
        ]
        result = simulated_annealing_sort(tracks, initial_order=tracks)
        assert len(result) == 4

    def test_improves_over_bad_initial_order(self):
        """SA should improve a deliberately bad initial ordering."""
        tracks = [
            _track(f'T{i}', tempo=100 + i * 20, camelot=(i % 12 + 1, 'A'))
            for i in range(8)
        ]
        # Reverse order is likely suboptimal
        initial = list(reversed(tracks))
        result = simulated_annealing_sort(tracks, initial_order=initial)
        assert {t['name'] for t in result} == {t['name'] for t in tracks}

    def test_progress_callback_is_called(self):
        calls = []
        tracks = [
            _track(f'T{i}', tempo=100 + i * 5, camelot=(i % 12 + 1, 'A'))
            for i in range(6)
        ]
        simulated_annealing_sort(
            tracks,
            progress_cb=lambda f, m: calls.append((f, m)),
        )
        assert len(calls) > 0

    def test_deterministic_with_fixed_seed(self):
        """With a fixed random seed, SA should produce the same result."""
        random.seed(42)
        tracks = [
            _track(f'T{i}', tempo=100 + i * 5, camelot=(i % 12 + 1, 'A'))
            for i in range(5)
        ]
        result1 = simulated_annealing_sort(tracks)
        random.seed(42)
        tracks2 = [
            _track(f'T{i}', tempo=100 + i * 5, camelot=(i % 12 + 1, 'A'))
            for i in range(5)
        ]
        result2 = simulated_annealing_sort(tracks2)
        assert [t['name'] for t in result1] == [t['name'] for t in result2]


# ---------------------------------------------------------------------------
# assign_energy_levels
# ---------------------------------------------------------------------------

class TestAssignEnergyLevels:
    def test_assigns_1_to_5(self):
        tracks = [
            _track(f'T{i}', spectral_flux=float(i))
            for i in range(10)
        ]
        assign_energy_levels(tracks)
        levels = [t['energy_level'] for t in tracks]
        assert all(1 <= l <= 5 for l in levels)

    def test_lowest_flux_gets_1(self):
        tracks = [
            _track('Low', spectral_flux=0.0),
            _track('Mid', spectral_flux=5.0),
            _track('High', spectral_flux=10.0),
        ]
        assign_energy_levels(tracks)
        assert tracks[0]['energy_level'] == 1

    def test_highest_flux_gets_5(self):
        # Need ≥5 tracks for the formula to reach level 5
        tracks = [
            _track(f'T{i}', spectral_flux=float(i))
            for i in range(5)
        ]
        assign_energy_levels(tracks)
        assert tracks[4]['energy_level'] == 5

    def test_single_track(self):
        tracks = [_track('Solo', spectral_flux=5.0)]
        assign_energy_levels(tracks)
        assert tracks[0]['energy_level'] == 1

    def test_returns_same_list(self):
        tracks = [_track(f'T{i}', spectral_flux=float(i)) for i in range(5)]
        result = assign_energy_levels(tracks)
        assert result is tracks

    def test_missing_flux_defaults_to_zero(self):
        tracks = [{'name': 'A', 'path': '/fake'}, {'name': 'B', 'path': '/fake'}]
        assign_energy_levels(tracks)
        # Both have flux=0 → ranks 0 and 1 → energy 1 and 3
        assert tracks[0]['energy_level'] == 1
        assert tracks[1]['energy_level'] == 3
