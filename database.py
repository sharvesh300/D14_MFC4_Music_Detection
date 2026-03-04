"""
database.py — Redis-backed persistence for songs and fingerprint hashes
=======================================================================

Key schema
----------
songs:counter           STRING  Auto-incrementing integer — the last assigned song id.
song:{id}               HASH    Song data: only "name" field (filename stem).
song:name:{name}        STRING  Maps a song's filename-stem → its integer id.
song:{id}:fingerprinted STRING  Presence flag; set after fingerprints are stored.
fp:{hash_value}         LIST    Entries "{song_id}:{time_offset}" for every indexed
                                 fingerprint that produced this hash value.

Public API
----------
get_connection()                         -> redis.Redis
create_database(r)                       -> None  (no-op, kept for compatibility)
insert_song(r, song_name)                -> int   (new song id)
insert_fingerprints_bulk(r, song_id, hashes)
match_fingerprints_bulk(r, hash_values)  -> [(hash_value, song_id, time_offset), …]
get_fingerprints(r, song_id)             -> [(hash_value, time_offset), …]
"""

import os
import redis

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB   = int(os.environ.get("REDIS_DB",   0))


def get_connection(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB):
    """Return a Redis client with decode_responses=True."""
    return redis.Redis(host=host, port=port, db=db, decode_responses=True,
                      socket_connect_timeout=5, socket_timeout=5)


def create_database(r):
    """No-op — Redis requires no schema creation. Kept for API compatibility."""
    pass


def insert_song(r, song_name):
    """
    Insert a song into Redis and return its integer id.

    Keys written:
        song:{id}        — Hash with a single "name" field.
        song:name:{name} — String mapping song name → id for reverse lookup.

    Parameters:
        r         : redis.Redis client
        song_name : base filename without extension
    """
    song_id = r.incr("songs:counter")
    r.hset(f"song:{song_id}", "name", song_name)
    r.set(f"song:name:{song_name}", song_id)
    return song_id


def insert_fingerprints_bulk(r, song_id, hashes):
    """
    Bulk-insert fingerprint hashes using a Redis pipeline.

    Each hash is appended to the list at key ``fp:{hash_value}``.
    List elements are packed as the string ``"{song_id}:{time_offset}"``.

    Parameters:
        r       : redis.Redis client
        song_id : integer song id
        hashes  : list of (hash_value: int, time_offset: int)
    """
    pipe = r.pipeline()
    for h, t in hashes:
        pipe.rpush(f"fp:{int(h)}", f"{int(song_id)}:{int(t)}")
    pipe.execute()


def match_fingerprints_bulk(r, hash_values):
    """
    Batch-lookup fingerprints for a list of hash values.

    Uses a single Redis pipeline round-trip (one LRANGE per unique hash).

    Parameters:
        r           : redis.Redis client
        hash_values : list of integer hash values

    Returns:
        list of (hash_value: int, song_id: int, time_offset: int)
    """
    if not hash_values:
        return []

    unique_hashes = list(set(int(h) for h in hash_values))

    pipe = r.pipeline()
    for hv in unique_hashes:
        pipe.lrange(f"fp:{hv}", 0, -1)
    raw_results = pipe.execute()

    results = []
    for hv, entries in zip(unique_hashes, raw_results):
        for entry in entries:
            sid_str, t_str = entry.split(":", 1)
            results.append((hv, int(sid_str), int(t_str)))
    return results


def get_fingerprints(r, song_id):
    """
    Not efficiently supported by the current list-per-hash key design.
    Kept for API compatibility — returns an empty list.
    """
    return []
