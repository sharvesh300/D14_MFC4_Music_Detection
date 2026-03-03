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
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT    NOT NULL
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


def insert_song(db_name, song_name):
    """Insert a song into the songs table and return its id."""
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    c.execute('''
        INSERT INTO songs (name) VALUES (?)
    ''', (song_name,))
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