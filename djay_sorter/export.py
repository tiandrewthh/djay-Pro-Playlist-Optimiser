"""Export — M3U, TSAF binary encoding, playlist clone, database backup."""

import datetime
import logging
import os
import shutil
import struct
import subprocess
import uuid

logger = logging.getLogger(__name__)

ROOT_PLAYLIST_UUID = '60526854-5D0D-47B8-9AA6-025C7516E7F4'
DJAY_DB = os.path.expanduser('~/Music/djay/djay Media Library.djayMediaLibrary/MediaLibrary.db')


def _tsaf_str(s):
    return b'\x08' + s.encode('utf-8') + b'\x00'


def encode_tsaf_playlist(pl_uuid, name):
    body = (
        _tsaf_str('ADCMediaItemPlaylist')
        + _tsaf_str(pl_uuid)            + _tsaf_str('uuid')
        + _tsaf_str(name)               + _tsaf_str('name')
        + _tsaf_str(ROOT_PLAYLIST_UUID) + _tsaf_str('parentUUID')
        + b'\x2e'                       + _tsaf_str('type')
        + b'\x00'
    )
    header = b'TSAF' + struct.pack('<HH', 3, 3) + struct.pack('<Q', 1) + struct.pack('<I', 8) + b'\x2b'
    return header + body


def encode_tsaf_playlist_item(item_uuid, pl_uuid, media_key_lowercase):
    body = (
        _tsaf_str('ADCMediaItemPlaylistItem')
        + _tsaf_str(item_uuid)            + _tsaf_str('uuid')
        + _tsaf_str(pl_uuid)              + _tsaf_str('playlistUUID')
        + _tsaf_str(media_key_lowercase)  + _tsaf_str('mediaItemUUID')
        + b'\x00'
    )
    header = b'TSAF' + struct.pack('<HH', 3, 3) + struct.pack('<Q', 0) + struct.pack('<I', 7) + b'\x2b'
    return header + body


def export_m3u(tracks, output_path):
    """Write an Extended M3U file for the given (sorted) track list."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('#EXTM3U\n')
        for t in tracks:
            artist = t.get('artist', '')
            name   = t.get('name', os.path.basename(t['path']))
            label  = f"{artist} - {name}" if artist else name
            f.write(f'#EXTINF:-1,{label}\n')
            f.write(f"{t['path']}\n")
    logger.info('M3U exported → %s', output_path)


def _backup_djay_db():
    """Checkpoint WAL then copy .db + .db-shm + .db-wal with a timestamp."""
    import sqlite3
    tmp = sqlite3.connect(DJAY_DB)
    tmp.execute('PRAGMA wal_checkpoint(FULL)')
    tmp.close()

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    for suffix in ('', '-shm', '-wal'):
        src = DJAY_DB + suffix
        if os.path.exists(src):
            shutil.copy2(src, DJAY_DB + f'.{ts}.bak{suffix}')
    return DJAY_DB + f'.{ts}.bak'


def create_sorted_clone(pl_rowid, sorted_tracks, new_name):
    """Create a brand-new djay Pro playlist as a sorted clone of an existing one."""
    import sqlite3

    # Safety: refuse to write if djay Pro is running
    check = subprocess.run(['pgrep', '-x', 'djay Pro'], capture_output=True)
    if check.returncode == 0:
        raise RuntimeError('djay Pro is running — quit it before writing the sorted playlist.')

    backup_path = _backup_djay_db()
    logger.info('Database backed up → %s', backup_path)

    db = sqlite3.connect(DJAY_DB)
    try:
        existing_media_rowids = {
            r[0] for r in db.execute('''
                SELECT r_media.dst
                FROM relationship_relationship r_item
                JOIN relationship_relationship r_media
                    ON r_media.src = r_item.src
                    AND r_media.name = "mediaItemPlaylistItemMediaItem"
                WHERE r_item.name = "mediaItemPlaylistItemPlaylist"
                  AND r_item.dst = ?
            ''', (pl_rowid,))
        }

        new_pl_uuid = str(uuid.uuid4()).upper()
        new_items   = [
            (str(uuid.uuid4()).upper(), t['_rowid'])
            for t in sorted_tracks
            if t['_rowid'] in existing_media_rowids
        ]
        if not new_items:
            raise ValueError('No valid tracks found in sorted list')

        pl_blob = encode_tsaf_playlist(new_pl_uuid, new_name)

        db.execute('BEGIN')

        # Insert new playlist
        db.execute(
            'INSERT INTO database2(collection, key, data) VALUES (?,?,?)',
            ('mediaItemPlaylists', new_pl_uuid, pl_blob),
        )
        new_pl_rowid = db.execute('SELECT last_insert_rowid()').fetchone()[0]

        db.execute(
            'INSERT INTO secondaryIndex_mediaItemPlaylistIndex(rowid, name) VALUES (?,?)',
            (new_pl_rowid, new_name),
        )

        # Playlist ↔ parent folder relationships
        mixes_rowid = db.execute(
            'SELECT rowid FROM secondaryIndex_mediaItemPlaylistIndex WHERE name="Mixes"'
        ).fetchone()[0]
        db.execute(
            'INSERT INTO relationship_relationship(name, src, dst, rules, manual) VALUES (?,?,?,?,?)',
            ('mediaItemPlaylistParent', new_pl_rowid, mixes_rowid, 4, 0),
        )
        db.execute(
            'INSERT INTO relationship_relationship(name, src, dst, rules, manual) VALUES (?,?,?,?,?)',
            ('mediaItemPlaylistChild', mixes_rowid, new_pl_rowid, 2, 0),
        )

        # Insert playlist items + relationships
        new_item_rowids = []
        for new_item_uuid, media_rowid in new_items:
            media_key = db.execute(
                'SELECT key FROM database2 WHERE rowid=?', (media_rowid,)
            ).fetchone()[0]
            item_blob = encode_tsaf_playlist_item(
                new_item_uuid, new_pl_uuid, media_key,
            )
            db.execute(
                'INSERT INTO database2(collection, key, data) VALUES (?,?,?)',
                ('mediaItemPlaylistItems', new_item_uuid, item_blob),
            )
            ir = db.execute('SELECT last_insert_rowid()').fetchone()[0]
            new_item_rowids.append(ir)

            db.execute(
                'INSERT INTO relationship_relationship(name, src, dst, rules, manual) VALUES (?,?,?,?,?)',
                ('mediaItemPlaylistItem', new_pl_rowid, ir, 2, 0),
            )
            db.execute(
                'INSERT INTO relationship_relationship(name, src, dst, rules, manual) VALUES (?,?,?,?,?)',
                ('mediaItemPlaylistItemPlaylist', ir, new_pl_rowid, 4, 0),
            )
            db.execute(
                'INSERT INTO relationship_relationship(name, src, dst, rules, manual) VALUES (?,?,?,?,?)',
                ('mediaItemPlaylistItemMediaItem', ir, media_rowid, 4, 0),
            )

        # Update view_mediaItemPlaylistsView
        page_key, page_data, page_count = db.execute(
            'SELECT pageKey, data, count FROM view_mediaItemPlaylistsView_page'
            ' WHERE "group"=?', (ROOT_PLAYLIST_UUID,)
        ).fetchone()
        db.execute(
            'UPDATE view_mediaItemPlaylistsView_page SET data=?, count=? WHERE pageKey=?',
            (page_data + struct.pack('<q', new_pl_rowid), page_count + 1, page_key),
        )
        db.execute(
            'INSERT INTO view_mediaItemPlaylistsView_map(rowid, pageKey) VALUES (?,?)',
            (new_pl_rowid, page_key),
        )

        # Create page in view_mediaItemPlaylistView
        new_view_page_key = str(uuid.uuid4()).upper()
        n = len(new_item_rowids)
        db.execute(
            'INSERT INTO view_mediaItemPlaylistView_page(pageKey, "group", prevPageKey, count, data)'
            ' VALUES (?,?,?,?,?)',
            (new_view_page_key, new_pl_uuid, None, n,
             b''.join(struct.pack('<q', r) for r in new_item_rowids)),
        )
        for ir in new_item_rowids:
            db.execute(
                'INSERT INTO view_mediaItemPlaylistView_map(rowid, pageKey) VALUES (?,?)',
                (ir, new_view_page_key),
            )

        # Update view_mediaView
        mv_page_key, mv_page_data, mv_count = db.execute(
            'SELECT pageKey, data, count FROM view_mediaView_page WHERE "group"=?',
            ('playlist',)
        ).fetchone()
        db.execute(
            'UPDATE view_mediaView_page SET data=?, count=? WHERE pageKey=?',
            (mv_page_data + struct.pack('<q', new_pl_rowid), mv_count + 1, mv_page_key),
        )
        db.execute(
            'INSERT INTO view_mediaView_map(rowid, pageKey) VALUES (?,?)',
            (new_pl_rowid, mv_page_key),
        )

        # FTS index entry
        db.execute(
            'INSERT INTO fts_searchIndex(docid, playlist) VALUES (?,?)',
            (new_pl_rowid, new_name),
        )

        # Integrity check before committing
        result = db.execute('PRAGMA integrity_check').fetchone()[0]
        if result != 'ok':
            raise ValueError(f'Integrity check failed: {result}')

        db.execute('COMMIT')
        logger.info("'%s' created in djay Pro (%d tracks).", new_name, len(new_item_rowids))
        logger.info('Relaunch djay Pro to see it.')

    except Exception as e:
        try:
            db.execute('ROLLBACK')
        except Exception:
            pass
        db.close()
        logger.error('Error: %s — changes rolled back, database unchanged.', e)
        raise

    db.close()
