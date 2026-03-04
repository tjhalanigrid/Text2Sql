"""
TERMINAL CHAT WITH DATABASE
Run:
python src/ask.py chinook_1
"""

import sys
from text2sql_engine import get_engine


# -------------------------------
# Pretty table printer
# -------------------------------
def print_table(cols, rows, limit=20):
    if not rows or not cols:
        print("No results\n")
        return

    cols = [str(c) for c in cols]

    widths = [max(len(c), 12) for c in cols]

    for r in rows[:limit]:
        for i, val in enumerate(r):
            widths[i] = max(widths[i], len(str(val)))

    header = " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols)))
    print("\n" + header)
    print("-" * len(header))

    for r in rows[:limit]:
        print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(cols))))

    if len(rows) > limit:
        print(f"\n... showing first {limit} rows of {len(rows)}")

    print()


# -------------------------------
# Main loop
# -------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python src/ask.py <db_id>")
        return

    db_id = sys.argv[1].strip()

    print("Loading model... (first time takes 20-40s)")
    engine = get_engine()

    print(f"\nConnected to database: {db_id}")
    print("Type 'exit' to quit\n")

    while True:
        try:
            q = input("Ask> ").strip()

            if not q:
                continue

            if q.lower() in ["exit", "quit"]:
                break

            result = engine.ask(q, db_id)

            if result is None:
                print("Model returned no output\n")
                continue

            print("\nGenerated SQL:")
            print(result.get("sql", "<no sql>"))

            if result.get("error"):
                print("\nSQL Error:")
                print(result["error"])
            else:
                print_table(
                    result.get("columns", []),
                    result.get("rows", []),
                )

        except KeyboardInterrupt:
            break
        except Exception as e:
            print("\nRuntime error:", e, "\n")

    print("\nBye!")


if __name__ == "__main__":
    main()