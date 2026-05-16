"""
DJ Playlist Optimiser — FastAPI wrapper
Run locally:  uvicorn api:app --reload
"""
from __future__ import annotations

import inspect
import io
import json
import threading
import uuid
import time
import asyncio
from typing import Any, Optional

import os
import sqlite3
import re
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from djay_sorter import (
    _build_cost_matrix,
    _build_ml_cost_matrix,
    assign_energy_levels,
    camelot_distance,
    create_sorted_clone,
    enrich_with_keys,
    get_playlist_tracks,
    greedy_sort,
    list_playlists,
    load_transition_model,
    ml_transition_cost,
    open_djay_db,
    simulated_annealing_sort,
    transition_cost,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title='DJ Playlist Optimiser',
    description='Optimise djay Pro playlist order using Camelot key + BPM + energy.',
    version='0.1.0',
)

_cors_origins_env = os.getenv('CORS_ORIGINS', '')
_cors_origins = (
    [o.strip() for o in _cors_origins_env.split(',') if o.strip()]
    if _cors_origins_env
    else ['http://localhost:5173', 'http://127.0.0.1:5173',
          'http://localhost:3000', 'http://127.0.0.1:3000']
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Sort is CPU-bound; run in a thread pool to avoid blocking the event loop
_STATIC = os.path.join(os.path.dirname(__file__), 'static')
_CONFIG_FILE = os.path.join(os.path.dirname(__file__), '.app_config.json')


def _load_app_config() -> dict:
    """Load persisted config from .app_config.json, or return defaults."""
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {'db_path': None}


def _save_app_config(config: dict) -> None:
    """Persist config to .app_config.json."""
    with open(_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


app_config = _load_app_config()
app.mount('/static', StaticFiles(directory=_STATIC), name='static')


@app.get('/', include_in_schema=False)
def root():
    return FileResponse(os.path.join(_STATIC, 'index.html'))


# ---------------------------------------------------------------------------
# Error-handling decorator
# ---------------------------------------------------------------------------

from functools import wraps


def _map_error(e: Exception) -> None:
    """Convert common exceptions to appropriate HTTP status codes."""
    if isinstance(e, HTTPException):
        raise
    if isinstance(e, ValueError):
        raise HTTPException(status_code=400, detail=str(e))
    if isinstance(e, FileNotFoundError):
        raise HTTPException(status_code=503, detail=str(e))
    if isinstance(e, sqlite3.OperationalError):
        if 'locked' in str(e).lower():
            raise HTTPException(status_code=503, detail='djay Pro database is locked. Please close djay Pro and try again.')
        raise HTTPException(status_code=500, detail=str(e))
    if isinstance(e, RuntimeError):
        raise HTTPException(status_code=409, detail=str(e))
    raise HTTPException(status_code=500, detail=str(e))


def handle_djay_errors(func):
    """Wrap endpoint logic to map common exceptions to HTTP status codes.

    Supports both sync and async (coroutine) endpoints.
    """
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                _map_error(e)
        return async_wrapper
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                _map_error(e)
        return wrapper


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Playlist(BaseModel):
    id: int
    name: str
    track_count: int


class Track(BaseModel):
    name: str
    artist: str
    bpm: float
    camelot: str
    energy_level: Optional[int]
    path: str


class SortRequest(BaseModel):
    playlist_id: int
    runs: int = Field(default=3, ge=1, le=20)
    max_bpm_jump: Optional[float] = None  # None = no limit


class ExportRequest(BaseModel):
    playlist_id: int
    output_name: str
    tracks: list[Track]


class ConfigRequest(BaseModel):
    db_path: Optional[str] = None


class SkipDetail(BaseModel):
    reason: str
    tracks: list[str]


class SortResponse(BaseModel):
    tracks: list[Track]
    greedy_cost: float
    sa_cost: float
    improvement_pct: float
    cost_function: str
    avg_bpm_jump: float
    max_bpm_jump: float
    key_clashes: int
    skipped_tracks: Optional[list[SkipDetail]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _do_sort(req: SortRequest, progress_cb=None) -> dict:
    """Blocking sort. progress_cb(fraction: float, message: str) is called throughout."""
    cb = progress_cb or (lambda f, m='': None)

    cb(0.0, 'Loading tracks…')
    db = None
    try:
        db = open_djay_db(custom_path=app_config['db_path'])
        tracks, skip_reasons = get_playlist_tracks(db, req.playlist_id)
    finally:
        if db:
            db.close()

    if not tracks:
        raise ValueError('No local tracks found in this playlist.')

    if len(tracks) < 2:
        raise ValueError('Playlist must contain at least 2 tracks to be sorted.')

    skipped_data = None
    if skip_reasons:
        # Group skips by reason
        reasons_map = {}
        for label, reason in skip_reasons:
            # Clean up reason text (e.g., "unsupported format (.wav)" -> "unsupported format")
            clean_reason = reason.split(' (')[0]
            reasons_map.setdefault(clean_reason, []).append(label)
        
        skipped_data = [
            {'reason': reason, 'tracks': labels} 
            for reason, labels in reasons_map.items()
        ]

    cb(0.05, f'Analysing {len(tracks)} tracks…')
    enriched = enrich_with_keys(
        tracks,
        progress_cb=lambda f, m: cb(0.05 + f * 0.15, m),
    )
    if not enriched:
        raise ValueError('No tracks could be analysed.')

    model = load_transition_model()
    if model is not None:
        cost_fn = lambda a, b: ml_transition_cost(a, b, model)
        cost_label = 'ml'
    else:
        cost_fn = transition_cost
        cost_label = 'heuristic'

    if req.max_bpm_jump is not None:
        _limit = req.max_bpm_jump
        _inner = cost_fn
        def cost_fn(a, b):  # noqa: F811
            # Use raw BPM diff — matches what is displayed in stats
            bpm_diff = abs(a['tempo'] - b['tempo'])
            return 100.0 if bpm_diff > _limit else _inner(a, b)

    cb(0.20, 'Building cost matrix…')
    n = len(enriched)
    # Use vectorised ML matrix when model is loaded — 5-10x faster for large playlists
    if model is not None:
        matrix = _build_ml_cost_matrix(enriched, model)
    else:
        matrix = _build_cost_matrix(enriched, cost_fn)

    cb(0.25, 'Greedy sort…')
    greedy = greedy_sort(enriched, cost_fn=cost_fn, _matrix=matrix)
    greedy_c = sum(cost_fn(greedy[i], greedy[i + 1]) for i in range(n - 1))

    best_sa, best_sa_c = None, float('inf')
    sa_start, sa_span = 0.30, 0.70
    for run_i in range(req.runs):
        cb(sa_start + sa_span * run_i / req.runs, f'SA run {run_i + 1}/{req.runs}…')
        run_base = sa_start + sa_span * run_i / req.runs
        run_size = sa_span / req.runs
        candidate = simulated_annealing_sort(
            enriched, initial_order=greedy, cost_fn=cost_fn, _matrix=matrix,
            progress_cb=lambda f, m: cb(run_base + f * run_size, m),
        )
        candidate_c = sum(cost_fn(candidate[i], candidate[i + 1]) for i in range(n - 1))
        if candidate_c < best_sa_c:
            best_sa, best_sa_c = candidate, candidate_c

    assign_energy_levels(best_sa)
    improvement = (greedy_c - best_sa_c) / greedy_c * 100 if greedy_c > 0 else 0.0

    if n > 1:
        bpm_jumps   = [abs(best_sa[i]['tempo'] - best_sa[i + 1]['tempo']) for i in range(n - 1)]
        key_clashes = sum(1 for i in range(n - 1)
                         if camelot_distance(best_sa[i]['camelot'], best_sa[i + 1]['camelot']) > 2)
        avg_bpm_jump = round(sum(bpm_jumps) / len(bpm_jumps), 1)
        max_bpm_jump = round(max(bpm_jumps), 1)
    else:
        key_clashes = 0
        avg_bpm_jump = max_bpm_jump = 0.0

    cb(1.0, 'Done!')

    tracks_out = []
    for t in best_sa:
        cam = f"{t['camelot'][0]}{t['camelot'][1]}"
        tracks_out.append({
            'name': t['name'],
            'artist': t.get('artist', ''),
            'bpm': t['tempo'],
            'camelot': cam,
            'energy_level': t.get('energy_level'),
            'path': t['path'],
        })

    return {
        'tracks': tracks_out,
        'greedy_cost': round(greedy_c, 4),
        'sa_cost': round(best_sa_c, 4),
        'improvement_pct': round(improvement, 1),
        'cost_function': cost_label,
        'avg_bpm_jump': avg_bpm_jump,
        'max_bpm_jump': max_bpm_jump,
        'key_clashes': key_clashes,
        'skipped_tracks': skipped_data,
    }


# ---------------------------------------------------------------------------
# Job tracking (for progress polling)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
JOB_CLEANUP_INTERVAL = 300   # 5 minutes
JOB_TTL              = 900   # 15 minutes


def _run_job(job_id: str, req: SortRequest) -> None:
    def cb(fraction: float, message: str = '') -> None:
        with _jobs_lock:
            _jobs[job_id]['progress'] = round(fraction, 3)
            _jobs[job_id]['stage']    = message

    try:
        result = _do_sort(req, progress_cb=cb)
        with _jobs_lock:
            _jobs[job_id].update({'status': 'done', 'progress': 1.0, 'result': result, 'finished_at': time.time()})
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id].update({'status': 'error', 'error': str(e), 'finished_at': time.time()})


def _cleanup_jobs():
    """Background worker to evict old jobs from memory to prevent leaks."""
    while True:
        try:
            time.sleep(JOB_CLEANUP_INTERVAL)
            now = time.time()
            with _jobs_lock:
                to_delete = [
                    jid for jid, data in _jobs.items()
                    if data.get('status') in ('done', 'error')
                    and now - data.get('finished_at', 0) > JOB_TTL
                ]
                for jid in to_delete:
                    del _jobs[jid]
        except Exception:
            # If anything goes wrong, sleep longer and retry — never let this thread die
            time.sleep(3600)


def _build_m3u(tracks: list[dict]) -> str:
    lines = ['#EXTM3U']
    for t in tracks:
        label = f"{t['artist']} - {t['name']}" if t['artist'] else t['name']
        lines.append(f"#EXTINF:-1,{label}")
        lines.append(t['path'])
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get('/config', summary='Get library configuration', description='Returns the current djay Pro database path and whether it was found on disk.')
def get_config():
    is_found = False
    db = None
    try:
        db = open_djay_db(custom_path=app_config['db_path'])
        is_found = True
    except Exception:
        is_found = False
    finally:
        if db:
            db.close()
    return {'db_path': app_config['db_path'], 'found': is_found}


@app.post('/config', summary='Set library configuration', description='Set a custom djay Pro database path. Persists across server restarts.')
@handle_djay_errors
def set_config(config: ConfigRequest):
    new_path = config.db_path
    if new_path and not os.path.exists(new_path):
        raise HTTPException(status_code=400, detail='The provided path does not exist on this machine.')
    app_config['db_path'] = new_path
    _save_app_config(app_config)
    return {'status': 'success', 'db_path': app_config['db_path']}


@app.get('/health', summary='Health check', description='Returns API status, database connectivity, and number of running sort jobs.')
def health():
    db = None
    try:
        db = open_djay_db(custom_path=app_config['db_path'])
        db_status = 'ok'
    except FileNotFoundError:
        db_status = 'not_found'
    except sqlite3.OperationalError:
        db_status = 'locked'
    except Exception:
        db_status = 'error'
    finally:
        if db:
            db.close()

    with _jobs_lock:
        running_jobs = sum(1 for j in _jobs.values() if j.get('status') == 'running')
    return {'status': 'ok', 'db': db_status, 'running_jobs': running_jobs}


@app.get('/playlists', response_model=list[Playlist], summary='List playlists', description='Returns all playlists from the djay Pro library with track counts.')
@handle_djay_errors
def get_playlists():
    db = open_djay_db(custom_path=app_config['db_path'])
    try:
        rows = list_playlists(db)
        return [{'id': r, 'name': n, 'track_count': c} for r, n, c in rows]
    finally:
        db.close()


@app.post('/sort', response_model=SortResponse, summary='Sort playlist (synchronous)', description='Sort a playlist using greedy + simulated annealing. Runs in a thread pool to avoid blocking the event loop. For long playlists, prefer /sort/start for progress feedback.')
@handle_djay_errors
async def sort_playlist(req: SortRequest):
    """Sort a playlist synchronously but offloaded to a thread pool.

    Unlike /sort/start this blocks until the sort is complete, but runs
    in a thread pool so it does NOT block the FastAPI event loop.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _do_sort, req)


@app.post('/sort/m3u', summary='Sort and export as M3U', description='Sort a playlist and return the result as a downloadable M3U file.')
@handle_djay_errors
async def sort_and_export_m3u(req: SortRequest):
    """Sort a playlist and return the result as a downloadable M3U file."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _do_sort, req)
    content = _build_m3u(result['tracks'])
    return StreamingResponse(
        io.BytesIO(content.encode('utf-8')),
        media_type='audio/x-mpegurl',
        headers={'Content-Disposition': 'attachment; filename="sorted_playlist.m3u"'},
    )


@app.post('/export', summary='Export sorted playlist to djay Pro', description='Write a previously sorted tracklist as a new playlist in the djay Pro library.')
@handle_djay_errors
def export_to_djay(req: ExportRequest):
    """Writes a previously sorted tracklist as a new playlist into djay Pro."""
    db = open_djay_db(custom_path=app_config['db_path'])
    try:
        playlist_tracks, _ = get_playlist_tracks(db, req.playlist_id)
    finally:
        db.close()

    path_to_rowid = {t['path']: t['_rowid'] for t in playlist_tracks}

    sorted_internal = []
    for t in req.tracks:
        rowid = path_to_rowid.get(t.path)
        if rowid is not None:
            sorted_internal.append({
                'name': t.name,
                'artist': t.artist,
                'path': t.path,
                'tempo': t.bpm,
                '_rowid': rowid,
            })

    if not sorted_internal:
        raise ValueError("No valid tracks from the sorted list could be found in the original playlist.")

    create_sorted_clone(req.playlist_id, sorted_internal, req.output_name)
    return {'status': 'success', 'message': f"Playlist '{req.output_name}' created."}


_MAX_CONCURRENT_JOBS = 3

@app.post('/sort/start', summary='Start sort job (async)', description='Start a background sort job. Returns a job_id to poll via /jobs/{job_id} for progress and results.')
@handle_djay_errors
def start_sort(req: SortRequest):
    """Start a sort job in the background; returns a job_id to poll."""
    with _jobs_lock:
        running = sum(1 for j in _jobs.values() if j.get('status') == 'running')
    if running >= _MAX_CONCURRENT_JOBS:
        raise HTTPException(status_code=429, detail='Too many sort jobs running. Please wait for one to finish.')

    # Guard against database locks before spawning a thread
    db = open_djay_db(custom_path=app_config['db_path'])
    db.close()

    job_id = uuid.uuid4().hex[:8]
    with _jobs_lock:
        _jobs[job_id] = {'status': 'running', 'progress': 0.0, 'stage': 'Starting…'}
    threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
    return {'job_id': job_id}


@app.get('/jobs/{job_id}', summary='Get sort job status', description='Poll a background sort job. Returns progress, stage message, and result when done.')
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return job

# Start cleanup thread on startup
threading.Thread(target=_cleanup_jobs, daemon=True).start()
