"""API endpoint tests — mock djay Pro DB, test FastAPI endpoints.

No real database or audio files required.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

import api
from api import app


@pytest.fixture()
def client():
    """Fresh test client with cleared job state."""
    api._jobs.clear()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helpers — realistic mock data
# ---------------------------------------------------------------------------

def _mock_track_dict(idx=1):
    return {
        '_rowid': idx,
        'name': f'Track {idx}',
        'artist': f'Artist {idx}',
        'path': f'/fake/track{idx}.mp3',
        'tempo': 120.0 + idx,
        'camelot': (idx % 12 + 1, 'A'),
        'spectral_flux': 5.0 + idx * 0.1,
        'spectral_centroid': 2000.0 + idx * 100,
        'onset_density': 1.5,
    }


PLAYLISTS = [(1, 'My Playlist', 3), (2, 'Empty Playlist', 0)]

TRACKS = [
    _mock_track_dict(1),
    _mock_track_dict(2),
    _mock_track_dict(3),
]

# What enrich_with_keys returns
ENRICHED = [
    {**t, 'key': 0, 'mode': 1, 'camelot': (8, 'B')} for t in TRACKS
]


def _mock_open_djay_db(custom_path=None):
    db = MagicMock()
    return db


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client):
        with patch('api.open_djay_db') as mock_db:
            mock_db.return_value = MagicMock()
            resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.json()
        assert data['status'] == 'ok'
        assert data['db'] == 'ok'
        assert data['running_jobs'] == 0

    def test_health_db_not_found(self, client):
        with patch('api.open_djay_db', side_effect=FileNotFoundError('no db')):
            resp = client.get('/health')
        data = resp.json()
        assert data['db'] == 'not_found'

    def test_health_db_locked(self, client):
        with patch('api.open_djay_db', side_effect=__import__('sqlite3').OperationalError('database is locked')):
            resp = client.get('/health')
        data = resp.json()
        assert data['db'] == 'locked'


# ---------------------------------------------------------------------------
# /config (GET + POST)
# ---------------------------------------------------------------------------

class TestConfig:
    def test_get_config(self, client):
        with patch('api.open_djay_db', return_value=MagicMock()):
            resp = client.get('/config')
        assert resp.status_code == 200
        assert 'db_path' in resp.json()
        assert 'found' in resp.json()

    def test_post_valid_db_path(self, tmp_path, client):
        fake_db = tmp_path / 'test.db'
        fake_db.touch()
        with patch.object(api, '_save_app_config', lambda cfg: None):
            resp = client.post('/config', json={'db_path': str(fake_db)})
        assert resp.status_code == 200
        data = resp.json()
        assert data['status'] == 'success'
        assert data['db_path'] == str(fake_db)

    def test_post_nonexistent_db_path(self, client):
        resp = client.post('/config', json={'db_path': '/no/such/path.db'})
        assert resp.status_code == 400

    def test_post_null_db_path(self, client):
        with patch.object(api, '_save_app_config', lambda cfg: None):
            resp = client.post('/config', json={'db_path': None})
        assert resp.status_code == 200
        assert resp.json()['db_path'] is None


# ---------------------------------------------------------------------------
# /playlists
# ---------------------------------------------------------------------------

class TestPlaylists:
    def test_list_playlists(self, client):
        with patch('api.open_djay_db', return_value=MagicMock()) as mock_db:
            with patch('api.list_playlists', return_value=PLAYLISTS):
                resp = client.get('/playlists')
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]['id'] == 1
        assert data[0]['name'] == 'My Playlist'
        assert data[0]['track_count'] == 3

    def test_playlists_db_not_found(self, client):
        with patch('api.open_djay_db', side_effect=FileNotFoundError('no db')):
            resp = client.get('/playlists')
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /sort (synchronous)
# ---------------------------------------------------------------------------

class TestSort:
    def _mock_sort_success(self, *args, **kwargs):
        """Mock _do_sort to return a realistic response."""
        return {
            'tracks': [
                {'name': t['name'], 'artist': t['artist'], 'bpm': t['tempo'],
                 'camelot': '8B', 'energy_level': i + 1, 'path': t['path']}
                for i, t in enumerate(ENRICHED)
            ],
            'greedy_cost': 0.45,
            'sa_cost': 0.32,
            'improvement_pct': 28.9,
            'cost_function': 'heuristic',
            'avg_bpm_jump': 2.0,
            'max_bpm_jump': 3.0,
            'key_clashes': 0,
            'skipped_tracks': None,
        }

    def test_sort_basic(self, client):
        with patch('api._do_sort', return_value=self._mock_sort_success()):
            resp = client.post('/sort', json={
                'playlist_id': 1,
                'runs': 2,
                'max_bpm_jump': 10.0,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data['tracks']) == 3
        assert data['cost_function'] == 'heuristic'
        assert data['sa_cost'] < data['greedy_cost']

    def test_sort_empty_playlist(self, client):
        with patch('api._do_sort', side_effect=ValueError('No local tracks')):
            resp = client.post('/sort', json={'playlist_id': 2})
        assert resp.status_code == 400

    def test_sort_single_track(self, client):
        with patch('api._do_sort', side_effect=ValueError('at least 2 tracks')):
            resp = client.post('/sort', json={'playlist_id': 1})
        assert resp.status_code == 400

    def test_sort_db_not_found(self, client):
        with patch('api._do_sort', side_effect=FileNotFoundError('db missing')):
            resp = client.post('/sort', json={'playlist_id': 1})
        assert resp.status_code == 503

    def test_sort_db_locked(self, client):
        import sqlite3
        with patch('api._do_sort', side_effect=sqlite3.OperationalError('database is locked')):
            resp = client.post('/sort', json={'playlist_id': 1})
        assert resp.status_code == 503
        assert 'locked' in resp.json()['detail'].lower()

    def test_sort_runs_validation(self, client):
        # runs=0 should fail (ge=1 constraint)
        resp = client.post('/sort', json={'playlist_id': 1, 'runs': 0})
        assert resp.status_code == 422

    def test_sort_cost_function_with_model(self, client):
        """When ML model is loaded, cost_function should be 'ml'."""
        result = self._mock_sort_success()
        result['cost_function'] = 'ml'
        with patch('api._do_sort', return_value=result):
            resp = client.post('/sort', json={'playlist_id': 1})
        assert resp.status_code == 200
        assert resp.json()['cost_function'] == 'ml'

    def test_sort_includes_skipped_tracks(self, client):
        result = self._mock_sort_success()
        result['skipped_tracks'] = [
            {'reason': 'Audio format not supported', 'tracks': ['Unknown Track']}
        ]
        with patch('api._do_sort', return_value=result):
            resp = client.post('/sort', json={'playlist_id': 1})
        assert resp.status_code == 200
        skips = resp.json()['skipped_tracks']
        assert len(skips) == 1
        assert skips[0]['reason'] == 'Audio format not supported'


# ---------------------------------------------------------------------------
# /sort/start + /jobs/{job_id} (async job polling)
# ---------------------------------------------------------------------------

class TestAsyncSort:
    def test_start_sort_job(self, client):
        with patch('api.open_djay_db', return_value=MagicMock()):
            with patch('api.threading.Thread') as mock_thread:
                resp = client.post('/sort/start', json={
                    'playlist_id': 1, 'runs': 2,
                })
        assert resp.status_code == 200
        data = resp.json()
        assert 'job_id' in data
        assert len(data['job_id']) == 8

    def test_get_job_not_found(self, client):
        resp = client.get('/jobs/nonexistent123')
        assert resp.status_code == 404

    def test_get_job_running(self, client):
        api._jobs['testjob1'] = {
            'status': 'running', 'progress': 0.35, 'stage': 'Analysing 5/20…',
        }
        resp = client.get('/jobs/testjob1')
        assert resp.status_code == 200
        data = resp.json()
        assert data['status'] == 'running'
        assert data['progress'] == 0.35
        assert 'Analysing' in data['stage']

    def test_get_job_done(self, client):
        api._jobs['testjob2'] = {
            'status': 'done', 'progress': 1.0, 'stage': 'Done!',
            'result': {'tracks': [], 'greedy_cost': 0.0, 'sa_cost': 0.0,
                      'improvement_pct': 0.0, 'cost_function': 'heuristic',
                      'avg_bpm_jump': 0.0, 'max_bpm_jump': 0.0, 'key_clashes': 0},
            'finished_at': 0,
        }
        resp = client.get('/jobs/testjob2')
        assert resp.status_code == 200
        data = resp.json()
        assert data['status'] == 'done'
        assert data['result'] is not None

    def test_get_job_error(self, client):
        api._jobs['testjob3'] = {
            'status': 'error', 'error': 'database is locked', 'finished_at': 0,
        }
        resp = client.get('/jobs/testjob3')
        assert resp.status_code == 200
        assert resp.json()['error'] == 'database is locked'

    def test_too_many_concurrent_jobs(self, client):
        """When 3 jobs are running, a 4th should be rejected with 429."""
        from api import _MAX_CONCURRENT_JOBS
        with patch('api.open_djay_db', return_value=MagicMock()):
            for i in range(_MAX_CONCURRENT_JOBS):
                api._jobs[f'busy{i}'] = {'status': 'running', 'progress': 0.1, 'stage': '…'}
            resp = client.post('/sort/start', json={'playlist_id': 1, 'runs': 1})
        assert resp.status_code == 429
        assert 'Too many' in resp.json()['detail']


# ---------------------------------------------------------------------------
# /sort/m3u (M3U export)
# ---------------------------------------------------------------------------

class TestM3UExport:
    def _mock_sort_result(self):
        return {
            'tracks': [
                {'name': 'A', 'artist': 'Art1', 'bpm': 120.0, 'camelot': '8B',
                 'energy_level': 1, 'path': '/fake/a.mp3'},
                {'name': 'B', 'artist': 'Art2', 'bpm': 122.0, 'camelot': '8B',
                 'energy_level': 2, 'path': '/fake/b.mp3'},
            ],
            'greedy_cost': 0.5, 'sa_cost': 0.3, 'improvement_pct': 40.0,
            'cost_function': 'heuristic', 'avg_bpm_jump': 2.0, 'max_bpm_jump': 2.0,
            'key_clashes': 0, 'skipped_tracks': None,
        }

    def test_m3u_export_returns_download(self, client):
        with patch('api._do_sort', return_value=self._mock_sort_result()):
            resp = client.post('/sort/m3u', json={'playlist_id': 1, 'runs': 1})
        assert resp.status_code == 200
        assert 'audio/x-mpegurl' in resp.headers.get('content-type', '')
        body = resp.text
        assert '#EXTM3U' in body
        assert '#EXTINF:-1,Art1 - A' in body
        assert '/fake/a.mp3' in body
        assert '/fake/b.mp3' in body

    def test_m3u_export_empty_playlist_name(self, client):
        result = self._mock_sort_result()
        result['tracks'][0]['artist'] = ''
        with patch('api._do_sort', return_value=result):
            resp = client.post('/sort/m3u', json={'playlist_id': 1})
        assert resp.status_code == 200
        assert '#EXTINF:-1,A' in resp.text  # artist-less tracks use name only


# ---------------------------------------------------------------------------
# /export (write to djay Pro)
# ---------------------------------------------------------------------------

class TestExportToDjay:
    def test_export_success(self, client):
        with patch('api.open_djay_db', return_value=MagicMock()):
            with patch('api.get_playlist_tracks', return_value=(TRACKS, [])):
                with patch('api.create_sorted_clone') as mock_clone:
                    resp = client.post('/export', json={
                        'playlist_id': 1,
                        'output_name': 'Sorted - My Playlist',
                        'tracks': [
                            {'name': 'Track 1', 'artist': 'Artist 1',
                             'bpm': 121.0, 'camelot': '8B', 'energy_level': 1,
                             'path': '/fake/track1.mp3'},
                        ],
                    })
        assert resp.status_code == 200
        data = resp.json()
        assert data['status'] == 'success'
        assert 'Sorted - My Playlist' in data['message']

    def test_export_no_matching_tracks(self, client):
        with patch('api.open_djay_db', return_value=MagicMock()):
            with patch('api.get_playlist_tracks', return_value=(TRACKS, [])):
                resp = client.post('/export', json={
                    'playlist_id': 1,
                    'output_name': 'Sorted',
                    'tracks': [
                        {'name': 'Unknown', 'artist': '', 'bpm': 120.0,
                         'camelot': '8B', 'energy_level': 1, 'path': '/fake/nonexistent.mp3'},
                    ],
                })
        assert resp.status_code == 400
