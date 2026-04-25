"""djay Pro database access — read playlists and track metadata."""

import logging
import os
import re
from urllib.parse import unquote

import sqlite3

from .constants import AUDIO_EXTS

logger = logging.getLogger(__name__)


def open_djay_db(custom_path=None):
    path = custom_path or os.path.expanduser(
        '~/Music/djay/djay Media Library.djayMediaLibrary/MediaLibrary.db'
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"djay Pro database not found at {path}")
    db = sqlite3.connect(path)
    db.execute('PRAGMA query_only = ON')
    return db


def list_playlists(db):
    rows = db.execute('''
        SELECT p.rowid, p.name, COUNT(r.src) as track_count
        FROM secondaryIndex_mediaItemPlaylistIndex p
        JOIN relationship_relationship r
            ON r.name = "mediaItemPlaylistItemPlaylist" AND r.dst = p.rowid
        GROUP BY p.rowid, p.name
        HAVING track_count > 0
        ORDER BY p.name
    ''').fetchall()
    return [(rowid, name, count) for rowid, name, count in rows]


def _track_label(media_rowid, fts_titles, path=None):
    title, artist = fts_titles.get(media_rowid, ('', ''))
    if title:
        return f"{artist} – {title}" if artist else title
    if path:
        return os.path.basename(path)
    return f"track #{media_rowid}"


def get_playlist_tracks(db, playlist_rowid):
    """Return (tracks, skip_reasons) for a playlist.

    Each track is a RawTrack dict (name, artist, path, tempo, _rowid).
    skip_reasons is a list of (label, reason) tuples.
    """
    rows = db.execute('''
        SELECT r_media.dst, idx.bpm
        FROM relationship_relationship r_item
        JOIN relationship_relationship r_media
            ON r_media.src = r_item.src
            AND r_media.name = "mediaItemPlaylistItemMediaItem"
        LEFT JOIN secondaryIndex_mediaItemIndex idx ON idx.rowid = r_media.dst
        WHERE r_item.name = "mediaItemPlaylistItemPlaylist"
          AND r_item.dst = ?
    ''', (playlist_rowid,)).fetchall()

    if not rows:
        return [], []

    media_rowids = [r[0] for r in rows]
    bpm_by_rowid = {r[0]: r[1] for r in rows}

    # Fetch only the keys for these rowids
    placeholders = ','.join('?' * len(media_rowids))
    rowid_to_key = dict(db.execute(
        f"SELECT rowid, key FROM database2 WHERE collection='mediaItems' AND rowid IN ({placeholders})",
        media_rowids,
    ))

    # Fetch file paths only for these keys
    keys = list(rowid_to_key.values())
    path_by_key = {}
    if keys:
        placeholders_k = ','.join('?' * len(keys))
        for key, data in db.execute(
            f"SELECT key, data FROM database2 WHERE collection='localMediaItemLocations' AND key IN ({placeholders_k})",
            keys,
        ):
            urls = re.findall(b'file:///[^\x00\x0a]+', data)
            if urls:
                path_by_key[key] = unquote(urls[0].decode().replace('file://', ''))

    # Fetch titles only for these rowids
    fts_titles = {}
    for rowid, title, artist in db.execute(
        f"SELECT rowid, c0title, c1artist FROM fts_searchIndex_content WHERE rowid IN ({placeholders})",
        media_rowids,
    ):
        fts_titles[rowid] = (title or '', artist or '')

    tracks = []
    skip_reasons = []
    for media_rowid in media_rowids:
        item_key = rowid_to_key.get(media_rowid)
        if not item_key:
            skip_reasons.append((_track_label(media_rowid, fts_titles), 'Track metadata not found in library'))
            continue
        path = path_by_key.get(item_key)
        if not path:
            skip_reasons.append((_track_label(media_rowid, fts_titles), 'No file path linked — remove and re-add to library'))
            continue
        if not os.path.exists(path):
            skip_reasons.append((_track_label(media_rowid, fts_titles, path), 'File moved or deleted on disk'))
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext not in AUDIO_EXTS:
            skip_reasons.append((_track_label(media_rowid, fts_titles, path), f'Audio format not supported ({ext.upper().lstrip(".")})'))
            continue

        title, artist = fts_titles.get(media_rowid, ('', ''))
        if not title:
            title = os.path.splitext(os.path.basename(path))[0]

        tracks.append({
            'name':    title,
            'artist':  artist,
            'path':    path,
            'tempo':   bpm_by_rowid.get(media_rowid) or 0.0,
            '_rowid':  media_rowid,
        })

    if skip_reasons:
        logger.warning('Skipped %d track(s):', len(skip_reasons))
        for label, reason in skip_reasons:
            logger.warning('  • %s — %s', label, reason)

    return tracks, skip_reasons
