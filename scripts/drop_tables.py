import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from database import get_connection


def main():
    r = get_connection()

    confirm = input(
        "This will DELETE all song and fingerprint data from Redis.\n"
        "Type 'yes' to confirm: "
    ).strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return

    deleted = 0
    cursor  = 0

    # Scan and delete all keys matching our schema patterns
    for pattern in ("song:*", "songs:counter", "fp:*"):
        while True:
            cursor, keys = r.scan(cursor=cursor, match=pattern, count=500)
            if keys:
                r.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        cursor = 0  # reset cursor for next pattern

    print(f"\nDeleted {deleted} key(s) from Redis. Database is now empty.")


if __name__ == "__main__":
    main()
