import sqlite3


def create_database(db_name):
    """Create the SQLite database, tables, and indices."""
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")

    # Songs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS songs (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            name                TEXT    NOT NULL,
            title               TEXT,
            artist              TEXT,
            album               TEXT,
            genre               TEXT,
            year                TEXT,
            track_number        TEXT,
            duration_seconds    REAL,
            duration_formatted  TEXT,
            sample_rate_hz      INTEGER,
            channels            INTEGER,
            bitrate_kbps        REAL,
            file_size_kb        REAL,
            cover_image_path    TEXT
        )
    ''')

    # Fingerprints table — hash_value BIGINT, time_offset INTEGER (frame index)
    c.execute('''
        CREATE TABLE IF NOT EXISTS fingerprints (
            song_id     INTEGER NOT NULL,
            hash_value  BIGINT  NOT NULL,
            time_offset INTEGER NOT NULL
        )
    ''')

    # Index for fast hash lookups
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_hash
        ON fingerprints(hash_value)
    ''')

    conn.commit()
    conn.close()


def insert_song(db_name, song_name, meta=None):
    """
    Insert a song into the songs table and return its id.

    Parameters:
        db_name   : path to the SQLite database
        song_name : base filename without extension (used as 'name')
        meta      : optional dict from extract_metadata(); all recognised
                    keys are stored in the corresponding columns.
    """
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    if meta:
        c.execute('''
            INSERT INTO songs (
                name, title, artist, album, genre, year, track_number,
                duration_seconds, duration_formatted,
                sample_rate_hz, channels, bitrate_kbps,
                file_size_kb, cover_image_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            song_name,
            meta.get("title") or song_name,
            meta.get("artist") or "",
            meta.get("album") or "",
            meta.get("genre") or "",
            meta.get("year") or "",
            meta.get("track_number") or "",
            meta.get("duration_seconds"),
            meta.get("duration_formatted") or "",
            meta.get("sample_rate_hz"),
            meta.get("channels"),
            meta.get("bitrate_kbps"),
            meta.get("file_size_kb"),
            meta.get("cover_image_path") or "",
        ))
    else:
        c.execute("INSERT INTO songs (name) VALUES (?)", (song_name,))

    song_id = c.lastrowid
    conn.commit()
    conn.close()
    return song_id


def insert_fingerprints_bulk(conn, song_id, hashes):
    """
    Bulk-insert fingerprint hashes using an existing open connection.

    Parameters:
        conn    : open sqlite3.Connection (caller manages lifecycle)
        song_id : integer song id
        hashes  : list of (hash_value: int, time_offset: int)
    """
    conn.executemany(
        "INSERT INTO fingerprints (song_id, hash_value, time_offset) VALUES (?, ?, ?)",
        [(song_id, int(h), int(t)) for h, t in hashes]
    )
    conn.commit()

def match_fingerprint(conn, hash_value):
    """
    Query for all (song_id, time_offset) rows matching a hash.

    Parameters:
        conn       : open sqlite3.Connection
        hash_value : integer hash to look up
    """
    c = conn.cursor()
    c.execute(
        "SELECT song_id, time_offset FROM fingerprints WHERE hash_value = ?",
        (int(hash_value),)
    )
    return c.fetchall()


def match_fingerprints_bulk(conn, hash_values):
    """
    Batch-query fingerprints for a list of hash values in a single SQL call.

    Returns list of (hash_value, song_id, time_offset).
    Chunks requests to stay under SQLite's 999-parameter limit.

    Parameters:
        conn        : open sqlite3.Connection
        hash_values : list of integer hash values
    """
    if not hash_values:
        return []

    results    = []
    chunk_size = 900

    for i in range(0, len(hash_values), chunk_size):
        chunk        = hash_values[i : i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT hash_value, song_id, time_offset FROM fingerprints "
            f"WHERE hash_value IN ({placeholders})",
            chunk
        ).fetchall()
        results.extend(rows)

    return results

def get_fingerprints(conn, song_id):
    """
    Retrieve all (hash_value, time_offset) rows for a song.

    Parameters:
        conn    : open sqlite3.Connection
        song_id : integer song id
    """
    c = conn.cursor()
    c.execute(
        "SELECT hash_value, time_offset FROM fingerprints WHERE song_id = ?",
        (song_id,)
    )
    return c.fetchall()