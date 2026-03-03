import os
import sys
import sqlite3

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "fingerprints.db")


def main():
    if not os.path.isfile(DB_PATH):
        print(f"Database not found: {DB_PATH}")
        return

    confirm = input("This will drop ALL tables. Type 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

        if not tables:
            print("No tables found.")
            return

        for (table,) in tables:
            if table.startswith("sqlite_"):
                continue
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            print(f"Dropped table: {table}")

        conn.commit()
        print("\nAll tables dropped.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
