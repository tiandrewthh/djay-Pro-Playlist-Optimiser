"""Integration tests — exercise the full sort pipeline with mocked dB/audio.

Mocks only the dB and audio-loading layers; the rest of _do_sort (cost matrix,
greedy sort, SA, energy levels, response serialisation) runs for real.
"""

import os
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from fastapi.testclient import TestClient

import api
from api import app


@pytest.fixture(autouse=True)
def clear_jobs():
    api._jobs.clear()
    yield


@pytest.fixture()
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Shared mock data
# ---------------------------------------------------------------------------

FAKE_DB_PATH = '/fake/djay.db'

MOCK_PLAYLISTS = [
    (1, 'House Mix', 4),
    (2, 'Techno Set', 3),
    (3, 'Tiny Session', 1),
]

# Raw tracks as returned by get_playlist_tracks
def _raw_tracks(count=4):
    return [
        {
            '_rowid': i + 1,
            'name': f'Track {i + 1}',
            'artist': f'Artist {i + 1}',
            'path': f'/fake/audio/track{i + 1}.mp3',
            'tempo': 120.0 + i * 5,
        }
        for i in range(count)
    ]

# Enriched tracks as returned by enrich_with_keys (simulating audio analysis)
def _enriched_tracks(raw_tracks):
    return [
        {
            **t,
            'key': i % 12,
            'mode': i % 2,
            'camelot': ((i % 12) + 1, 'A' if i % 2 else 'B'),
            'spectral_flux': 3.0 + i * 0.5,
            'spectral_centroid': 1800 + i * 200,
            'onset_density': 1.0 + i * 0.3,
        }
        for i, t in enumerate(raw_tracks)
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_open_djay_db():
    """Patch open_djay_db to return a mock connection."""
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchall.return_value = []
    return patch('api.open_djay_db', return_value=mock_db)


def _setup_get_playlist_tracks(raw_tracks, skip_reasons=None):
    """Patch get_playlist_tracks to return fixed track data."""
    return patch('api.get_playlist_tracks', return_value=(raw_tracks, skip_reasons or []))


def _setup_enrich_with_keys(raw_tracks, enriched):
    """Patch enrich_with_keys to return pre-baked enriched tracks."""
    def fake_enrich(tracks, progress_cb=None):
        if progress_cb:
            progress_cb(1.0, 'Analysing complete')
        return enriched
    return patch('api.enrich_with_keys', side_effect=fake_enrich)


# ---------------------------------------------------------------------------
# /sort — full pipeline (no _do_sort mock)
# ---------------------------------------------------------------------------

class TestSortIntegration:
    """Test /sort endpoint with the full _do_sort pipeline running."""

    def test_full_sort_pipeline_heuristic(self, client):
        """End-to-end sort: load tracks → enrich → cost matrix → greedy → SA → response."""
        raw = _raw_tracks(4)
        enriched = _enriched_tracks(raw)

        with _setup_open_djay_db():
            with _setup_get_playlist_tracks(raw):
                with _setup_enrich_with_keys(raw, enriched):
                    with patch('api.load_transition_model', return_value=None):
                        resp = client.post('/sort', json={
                            'playlist_id': 1,
                            'runs': 2,
                            'max_bpm_jump': 15.0,
                        })

        assert resp.status_code == 200
        data = resp.json()

        # Response shape
        assert len(data['tracks']) == 4
        assert 'greedy_cost' in data
        assert 'sa_cost' in data
        assert 'improvement_pct' in data
        assert 'cost_function' in data
        assert 'avg_bpm_jump' in data
        assert 'max_bpm_jump' in data
        assert 'key_clashes' in data

        # Heuristic (no ML model)
        assert data['cost_function'] == 'heuristic'

        # Each track has required fields
        for t in data['tracks']:
            assert all(k in t for k in ('name', 'artist', 'bpm', 'camelot', 'energy_level', 'path'))
            assert t['camelot'] in {f'{n}{l}' for n in range(1, 13) for l in 'AB'}
            assert 1 <= t['energy_level'] <= 5

    def test_full_sort_pipeline_with_ml_model(self, client):
        """Sort using ML transition model — should use vectorised cost matrix."""
        raw = _raw_tracks(3)
        enriched = _enriched_tracks(raw)

        mock_model = MagicMock()
        # predict_proba must handle N×N feature array from _build_ml_cost_matrix
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])  # [bad, good]

        with _setup_open_djay_db():
            with _setup_get_playlist_tracks(raw):
                with _setup_enrich_with_keys(raw, enriched):
                    with patch('api.load_transition_model', return_value=mock_model):
                        resp = client.post('/sort', json={
                            'playlist_id': 1,
                            'runs': 1,
                        })

        assert resp.status_code == 200
        data = resp.json()
        assert data['cost_function'] == 'ml'
        assert len(data['tracks']) == 3

    def test_sort_progress_callback_called(self, client):
        """Verify progress callbacks are invoked throughout the pipeline."""
        raw = _raw_tracks(3)
        enriched = _enriched_tracks(raw)
        stages_called = []

        original_do_sort = api._do_sort

        def captured_do_sort(req, progress_cb=None):
            def wrapped_cb(f, m=''):
                stages_called.append((f, m))
            return original_do_sort(req, progress_cb=wrapped_cb)

        with _setup_open_djay_db():
            with _setup_get_playlist_tracks(raw):
                with _setup_enrich_with_keys(raw, enriched):
                    with patch('api.load_transition_model', return_value=None):
                        with patch('api._do_sort', side_effect=captured_do_sort):
                            resp = client.post('/sort', json={
                                'playlist_id': 1,
                                'runs': 1,
                            })

        assert resp.status_code == 200
        assert len(stages_called) > 0
        # Should have stage labels
        labels = [s[1] for s in stages_called if s[1]]
        assert any('Loading' in l for l in labels), f'Missing loading stage: {labels}'
        assert any('Done' in l for l in labels), f'Missing done stage: {labels}'

    def test_sort_no_tracks(self, client):
        """Should return 400 when playlist has no local tracks."""
        with _setup_open_djay_db():
            with _setup_get_playlist_tracks([], skip_reasons=[('Removed Track', 'Track metadata not found')]):
                with patch('api.load_transition_model', return_value=None):
                    resp = client.post('/sort', json={'playlist_id': 99})

        assert resp.status_code == 400
        assert 'No local tracks' in resp.json()['detail']

    def test_sort_single_track(self, client):
        """Should return 400 for playlists with only 1 playable track."""
        raw = _raw_tracks(1)
        enriched = _enriched_tracks(raw)

        with _setup_open_djay_db():
            with _setup_get_playlist_tracks(raw):
                with _setup_enrich_with_keys(raw, enriched):
                    with patch('api.load_transition_model', return_value=None):
                        resp = client.post('/sort', json={'playlist_id': 3})

        assert resp.status_code == 400

    def test_sort_all_tracks_fail_analysis(self, client):
        """Should return 400 when analysis fails for every track."""
        raw = _raw_tracks(3)

        def failing_enrich(tracks, progress_cb=None):
            return []

        with _setup_open_djay_db():
            with _setup_get_playlist_tracks(raw):
                with patch('api.enrich_with_keys', side_effect=failing_enrich):
                    with patch('api.load_transition_model', return_value=None):
                        resp = client.post('/sort', json={'playlist_id': 1})

        assert resp.status_code == 400
        assert 'analysed' in resp.json()['detail'].lower()

    def test_sort_preserves_bpm_from_djay(self, client):
        """Should use djay Pro BPM rather than librosa detection."""
        raw = _raw_tracks(3)
        raw[0]['tempo'] = 128.5  # djay Pro BPM
        enriched = _enriched_tracks(raw)
        enriched[0]['detected_bpm'] = 999.0  # wrong detection should be ignored

        with _setup_open_djay_db():
            with _setup_get_playlist_tracks(raw):
                with _setup_enrich_with_keys(raw, enriched):
                    with patch('api.load_transition_model', return_value=None):
                        resp = client.post('/sort', json={'playlist_id': 1})

        assert resp.status_code == 200
        data = resp.json()
        djay_bpm = data['tracks'][0]['bpm']
        assert djay_bpm == 128.5, f'Expecting djay BPM 128.5, got {djay_bpm}'

    def test_sort_with_max_bpm_jump_constraint(self, client):
        """Should apply BPM jump constraint and report it in stats.

        The constraint is a heavy penalty (100.0) for transitions exceeding the limit,
        but SA may still use one if overall cost is better. We verify stats are
        reported correctly, not that max_bpm_jump is capped.
        """
        raw = _raw_tracks(5)
        enriched = _enriched_tracks(raw)

        with _setup_open_djay_db():
            with _setup_get_playlist_tracks(raw):
                with _setup_enrich_with_keys(raw, enriched):
                    with patch('api.load_transition_model', return_value=None):
                        resp = client.post('/sort', json={
                            'playlist_id': 1,
                            'runs': 1,
                            'max_bpm_jump': 3.0,  # tight constraint
                        })

        assert resp.status_code == 200
        data = resp.json()
        # Stats should be reported regardless of whether constraint is satisfied
        assert 'max_bpm_jump' in data
        assert 'avg_bpm_jump' in data
        assert data['sa_cost'] <= data['greedy_cost']

    def test_sa_cost_less_than_greedy_cost(self, client):
        """Simulated annealing should improve or match the initial greedy solution."""
        raw = _raw_tracks(6)
        enriched = _enriched_tracks(raw)

        with _setup_open_djay_db():
            with _setup_get_playlist_tracks(raw):
                with _setup_enrich_with_keys(raw, enriched):
                    with patch('api.load_transition_model', return_value=None):
                        # Use more SA runs to increase chance of improvement
                        resp = client.post('/sort', json={
                            'playlist_id': 1,
                            'runs': 5,
                        })

        assert resp.status_code == 200
        data = resp.json()
        assert data['sa_cost'] <= data['greedy_cost']
        # Improvement should be reported (can be 0)
        assert data['improvement_pct'] >= 0.0


# ---------------------------------------------------------------------------
# /sort/m3u — full pipeline + M3U export
# ---------------------------------------------------------------------------

class TestM3UExportIntegration:
    """Test /sort/m3u with full sort pipeline and M3U generation."""

    def test_m3u_full_pipeline(self, client):
        """Full sort then M3U export."""
        raw = _raw_tracks(3)
        enriched = _enriched_tracks(raw)

        with _setup_open_djay_db():
            with _setup_get_playlist_tracks(raw):
                with _setup_enrich_with_keys(raw, enriched):
                    with patch('api.load_transition_model', return_value=None):
                        resp = client.post('/sort/m3u', json={
                            'playlist_id': 1,
                            'runs': 1,
                        })

        assert resp.status_code == 200
        assert 'audio/x-mpegurl' in resp.headers.get('content-type', '')

        # M3U content validation
        body = resp.text
        assert body.startswith('#EXTM3U')
        lines = body.strip().split('\n')
        # Each track = 1 #EXTINF + 1 path, plus header
        assert len(lines) == 1 + len(raw) * 2
        for t in raw:
            assert t['path'] in body

    def test_m3u_with_artist_and_title(self, client):
        """M3U should include artist - title in EXTINF."""
        raw = _raw_tracks(2)
        enriched = _enriched_tracks(raw)

        with _setup_open_djay_db():
            with _setup_get_playlist_tracks(raw):
                with _setup_enrich_with_keys(raw, enriched):
                    with patch('api.load_transition_model', return_value=None):
                        resp = client.post('/sort/m3u', json={
                            'playlist_id': 1,
                            'runs': 1,
                        })

        assert resp.status_code == 200
        assert '#EXTINF:-1,Artist 1 - Track 1' in resp.text


# ---------------------------------------------------------------------------
# /export — full pipeline → export to djay Pro
# ---------------------------------------------------------------------------

class TestExportIntegration:
    """Test /export endpoint: fetch original → reorder → create_sorted_clone."""

    def test_export_after_sort(self, client):
        """Full export flow: get originals, match by path, call create_sorted_clone."""
        raw = _raw_tracks(3)

        with _setup_open_djay_db():
            with _setup_get_playlist_tracks(raw):
                with patch('api.create_sorted_clone') as mock_clone:
                    resp = client.post('/export', json={
                        'playlist_id': 1,
                        'output_name': 'Sorted - House Mix',
                        'tracks': [
                            {
                                'name': 'Track 2',
                                'artist': 'Artist 2',
                                'bpm': 125.0,
                                'camelot': '3A',
                                'energy_level': 2,
                                'path': '/fake/audio/track2.mp3',
                            },
                            {
                                'name': 'Track 1',
                                'artist': 'Artist 1',
                                'bpm': 120.0,
                                'camelot': '2B',
                                'energy_level': 1,
                                'path': '/fake/audio/track1.mp3',
                            },
                        ],
                    })

        assert resp.status_code == 200
        data = resp.json()
        assert 'Sorted - House Mix' in data['message']

        # create_sorted_clone should receive the reordered tracks
        mock_clone.assert_called_once()
        call_args = mock_clone.call_args
        assert call_args[0][0] == 1  # playlist_id
        assert call_args[0][2] == 'Sorted - House Mix'  # output_name
        sorted_tracks = call_args[0][1]
        assert len(sorted_tracks) == 2
        assert sorted_tracks[0]['name'] == 'Track 2'  # reordered

    def test_export_partial_match(self, client):
        """When only some sorted tracks match original paths, only matched tracks are exported."""
        raw = _raw_tracks(2)

        with _setup_open_djay_db():
            with _setup_get_playlist_tracks(raw):
                with patch('api.create_sorted_clone') as mock_clone:
                    resp = client.post('/export', json={
                        'playlist_id': 1,
                        'output_name': 'Partial',
                        'tracks': [
                            {
                                'name': 'Track 1',
                                'artist': 'Artist 1',
                                'bpm': 120.0,
                                'camelot': '2B',
                                'energy_level': 1,
                                'path': '/fake/audio/track1.mp3',  # matches
                            },
                            {
                                'name': 'Ghost Track',
                                'artist': '',
                                'bpm': 130.0,
                                'camelot': '5A',
                                'energy_level': 3,
                                'path': '/fake/audio/nonexistent.mp3',  # doesn't match
                            },
                        ],
                    })

        assert resp.status_code == 200
        sorted_tracks = mock_clone.call_args[0][1]
        assert len(sorted_tracks) == 1
        assert sorted_tracks[0]['name'] == 'Track 1'

    def test_export_no_matches(self, client):
        """When no sorted tracks match original paths, should return 400."""
        raw = _raw_tracks(2)

        with _setup_open_djay_db():
            with _setup_get_playlist_tracks(raw):
                resp = client.post('/export', json={
                    'playlist_id': 1,
                    'output_name': 'Fail',
                    'tracks': [
                        {
                            'name': 'Unknown',
                            'artist': '',
                            'bpm': 120.0,
                            'camelot': '2B',
                            'energy_level': 1,
                            'path': '/fake/audio/no-match.mp3',
                        },
                    ],
                })

        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /health — real dB connectivity
# ---------------------------------------------------------------------------

class TestHealthIntegration:
    """Test /health with various dB states."""

    def test_db_connects_and_queries(self, client, tmp_path):
        """When a real SQLite DB with expected schema exists, health check passes."""
        db_path = tmp_path / 'test.db'
        conn = __import__('sqlite3').connect(str(db_path))
        # Minimal tables to satisfy open_djay_db checks
        conn.execute('CREATE TABLE secondaryIndex_mediaItemPlaylistIndex (rowid INTEGER, name TEXT)')
        conn.close()

        with patch.dict(api.app_config, {'db_path': str(db_path)}):
            resp = client.get('/health')

        assert resp.status_code == 200
        assert resp.json()['db'] == 'ok'

    def test_db_file_not_found(self, client):
        with patch('api.open_djay_db', side_effect=FileNotFoundError('no db')):
            resp = client.get('/health')
        data = resp.json()
        assert data['db'] == 'not_found'
        assert data['status'] == 'ok'  # API itself is healthy

    def test_db_locked(self, client):
        import sqlite3
        with patch('api.open_djay_db', side_effect=sqlite3.OperationalError('database is locked')):
            resp = client.get('/health')
        data = resp.json()
        assert data['db'] == 'locked'


# ---------------------------------------------------------------------------
# /config persistence (reads from disk)
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Test /config persistence across requests."""

    def test_config_survives_restart(self, client, tmp_path):
        """Config written to disk should be read back on next request."""
        config_file = tmp_path / '.app_config.json'
        config_file.write_text(json.dumps({'db_path': '/tmp/custom.db'}))

        with patch.object(api, '_CONFIG_FILE', str(config_file)):
            # Reload config
            api.app_config = json.loads(config_file.read_text())

            resp = client.get('/config')

        assert resp.status_code == 200
        assert resp.json()['db_path'] == '/tmp/custom.db'
