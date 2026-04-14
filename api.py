"""
DJ Playlist Optimiser — FastAPI wrapper
Run locally:  uvicorn api:app --reload
"""
from __future__ import annotations

import io
import threading
import uuid
from typing import Any, Optional

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from djay_sorter import (
    _build_cost_matrix,
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173',
                   'http://localhost:3000', 'http://127.0.0.1:3000'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Sort is CPU-bound; run in a thread pool to avoid blocking the event loop
_STATIC = os.path.join(os.path.dirname(__file__), 'static')

app_config = {
    'db_path': None,
}
app.mount('/static', StaticFiles(directory=_STATIC), name='static')


@app.get('/', include_in_schema=False)
def root():
    return FileResponse(os.path.join(_STATIC, 'index.html'))


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
    write: bool = False
    output_name: Optional[str] = None
    max_bpm_jump: Optional[float] = None  # None = no limit


class SortResponse(BaseModel):
    tracks: list[Track]
    greedy_cost: float
    sa_cost: float
    improvement_pct: float
    cost_function: str
    avg_bpm_jump: float
    max_bpm_jump: float
    key_clashes: int
    skipped_message: Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _do_sort(req: SortRequest, progress_cb=None) -> dict:
    """Blocking sort. progress_cb(fraction: float, message: str) is called throughout."""
    cb = progress_cb or (lambda f, m='': None)

    cb(0.0, 'Loading tracks…')
    db = open_djay_db(custom_path=app_config['db_path'])
    tracks, skip_reasons = get_playlist_tracks(db, req.playlist_id)
    playlists = list_playlists(db)
    pl_name = next((n for r, n, _ in playlists if r == req.playlist_id), 'Sorted')
    db.close()

    if not tracks:
        raise ValueError('No local tracks found in this playlist.')

    if skip_reasons:
        from collections import Counter
        counts = Counter(skip_reasons)
        parts = [f"{reason} ×{n}" if n > 1 else reason for reason, n in counts.most_common()]
        skipped_msg = f"{len(skip_reasons)} track{'s' if len(skip_reasons) > 1 else ''} skipped — {', '.join(parts)}"
    else:
        skipped_msg = None

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

    if req.write:
        cb(0.98, 'Writing to djay Pro…')
        output_name = req.output_name or f'{pl_name} (Sorted)'
        create_sorted_clone(req.playlist_id, best_sa, output_name)

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
        'skipped_message': skipped_msg,
    }


# ---------------------------------------------------------------------------
# Job tracking (for progress polling)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}


def _run_job(job_id: str, req: SortRequest) -> None:
    def cb(fraction: float, message: str = '') -> None:
        _jobs[job_id]['progress'] = round(fraction, 3)
        _jobs[job_id]['stage']    = message

    try:
        result = _do_sort(req, progress_cb=cb)
        _jobs[job_id].update({'status': 'done', 'progress': 1.0, 'result': result})
    except Exception as e:
        _jobs[job_id].update({'status': 'error', 'error': str(e)})


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

@app.get('/config')
def get_config():
    is_found = False
    try:
        db = open_djay_db(custom_path=app_config['db_path'])
        db.close()
        is_found = True
    except Exception:
        is_found = False
    return {'db_path': app_config['db_path'], 'found': is_found}


@app.post('/config')
def set_config(config: dict):
    new_path = config.get('db_path')
    if new_path and not os.path.exists(new_path):
        raise HTTPException(status_code=400, detail='The provided path does not exist on this machine.')
    app_config['db_path'] = new_path
    return {'status': 'success', 'db_path': app_config['db_path']}


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/playlists', response_model=list[Playlist])
def get_playlists():
    try:
        db = open_djay_db(custom_path=app_config['db_path'])
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    rows = list_playlists(db)
    db.close()
    return [{'id': r, 'name': n, 'track_count': c} for r, n, c in rows]


@app.post('/sort', response_model=SortResponse)
def sort_playlist(req: SortRequest):
    try:
        return _do_sort(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post('/sort/m3u')
def sort_and_export_m3u(req: SortRequest):
    """Sort a playlist and return the result as a downloadable M3U file."""
    try:
        result = _do_sort(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    content = _build_m3u(result['tracks'])
    return StreamingResponse(
        io.BytesIO(content.encode('utf-8')),
        media_type='audio/x-mpegurl',
        headers={'Content-Disposition': 'attachment; filename="sorted_playlist.m3u"'},
    )


@app.post('/sort/start')
def start_sort(req: SortRequest):
    """Start a sort job in the background; returns a job_id to poll."""
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {'status': 'running', 'progress': 0.0, 'stage': 'Starting…'}
    threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
    return {'job_id': job_id}


@app.get('/jobs/{job_id}')
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return job
