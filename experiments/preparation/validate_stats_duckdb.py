#!/usr/bin/env python3
"""Independent DuckDB COUNT(*) oracle for the fixed STATS queries."""

import argparse
import json
from pathlib import Path



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--query-manifest", required=True, type=Path)
    parser.add_argument("--temp-dir", required=True, type=Path)
    args = parser.parse_args()
    import duckdb

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    query_manifest = json.loads(args.query_manifest.read_text(encoding="utf-8"))
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(":memory:")
    connection.execute("PRAGMA threads=1")
    connection.execute("SET memory_limit='3GB'")
    escaped_temp = str(args.temp_dir).replace("'", "''")
    connection.execute(f"SET temp_directory='{escaped_temp}'")
    type_map = {"int": "BIGINT", "timestamp": "TIMESTAMP"}
    for item in source_manifest["files"]:
        if item["role"] != "table":
            continue
        columns = ", ".join(
            f"'{name}':'{type_map[kind]}'" for name, kind in item["columns"].items()
        )
        path = str(args.raw_dir / item["name"]).replace("'", "''")
        table = item["table"].replace('"', '""')
        connection.execute(
            f'CREATE VIEW "{table}" AS SELECT * FROM read_csv(\'{path}\', '
            f"header=true, columns={{ {columns} }}, nullstr='')"
        )
    counts = {}
    for query in query_manifest["queries"]:
        counts[query["id"]] = str(int(connection.execute(query["count_sql"]).fetchone()[0]))
    connection.close()
    print(json.dumps({"duckdb_version": duckdb.__version__, "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
