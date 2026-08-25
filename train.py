"""
Training script for Vanna — loads DDL and domain documentation into ChromaDB.

Usage:
    python train.py                      # Load from all configured sources
    python train.py --postgres-only      # Load only PostgreSQL schemas
    python train.py --bigquery-only      # Load only BigQuery schemas
    python train.py --schemas bronze,gold  # Override TRAIN_SCHEMAS for this run
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv

from vanna.integrations.chromadb.agent_memory import ChromaAgentMemory
from vanna.core.tool import ToolContext
from vanna.core.user import User

load_dotenv()

# Domain notes that don't belong to any single table - competition/table-naming
# knowledge an LLM can't infer from DDL alone. Mirrors the equivalent notes in
# the sibling basketball-gpt app's query_engine.py, since both query the same
# sportsanalytics database.
DOMAIN_NOTES = [
    "Table prefixes indicate competitions: b_el = EuroLeague, b_ec = EuroCup, "
    "b_cl = Champions League, b_bbl = Basketball Bundesliga. boxscore tables are "
    "usually best for player/team game totals and rankings; playbyplay tables for "
    "event sequences and possession-level questions; player_info tables for player "
    "lookup and roster attributes. b_bbl_boxscore uses date_final, home_team_final, "
    "away_team_final instead of date, home_team, away_team used by the other three "
    "leagues' boxscore tables.",
    "bronze.* boxscore/playbyplay tables have NO season column, only a per-game date "
    "(date_final for BBL). A competition season 'YYYY-(YYYY+1)' runs from around "
    "August of YYYY to around July of YYYY+1, crossing the calendar-year boundary - "
    "a question about 'season 2025-2026' or 'season 2025' needs a filter like "
    "date >= '2025-08-01' AND date < '2026-08-01', never a calendar-year filter "
    "(date BETWEEN '2025-01-01' AND '2025-12-31'), which silently cuts the season "
    "in half. gold.g_el_players, g_ec_players, g_cl_players, and g_bbl_players "
    "already have a saison column formatted like '2025-2026' - prefer filtering "
    "there over date math whenever the requested stat exists in a gold table.",
    "When grouping player stats by season, GROUP BY player_name alone, not "
    "(player_name, team). Several teams have mid-season sponsor renames (the same "
    "player then has two team-name rows in one season), which silently splits and "
    "undercounts a renamed team's players if team is in the GROUP BY.",
]


def get_dummy_context(memory: ChromaAgentMemory) -> ToolContext:
    """Create a minimal ToolContext for training operations."""
    user = User(id="trainer", username="trainer", group_memberships=["admin"])
    return ToolContext(
        user=user,
        conversation_id="training",
        request_id="training",
        agent_memory=memory,
        metadata={},
    )


def get_postgres_ddl(connection_string: str, schemas: list[str]) -> list[dict]:
    """Extract table DDL from PostgreSQL via information_schema, scoped to `schemas`."""
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(connection_string)
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Only the schemas explicitly named - not "every non-system schema", which
    # would happily pull in public or any future unrelated schema too.
    cursor.execute("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema = ANY(%s)
          AND table_type = 'BASE TABLE'
        ORDER BY table_schema, table_name
    """, (schemas,))
    tables = cursor.fetchall()

    ddl_entries = []
    for table in tables:
        schema = table["table_schema"]
        name = table["table_name"]
        full_name = f"{schema}.{name}" if schema != "public" else name

        # Get column definitions
        cursor.execute("""
            SELECT column_name, data_type, is_nullable, column_default,
                   character_maximum_length, numeric_precision
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """, (schema, name))
        columns = cursor.fetchall()

        # Build DDL string
        col_defs = []
        for col in columns:
            col_def = f"  {col['column_name']} {col['data_type']}"
            if col["character_maximum_length"]:
                col_def += f"({col['character_maximum_length']})"
            if col["is_nullable"] == "NO":
                col_def += " NOT NULL"
            if col["column_default"]:
                col_def += f" DEFAULT {col['column_default']}"
            col_defs.append(col_def)

        ddl = f"CREATE TABLE {full_name} (\n" + ",\n".join(col_defs) + "\n);"

        ddl_entries.append({
            "content": f"DDL for table {full_name}:\n{ddl}",
            "table": full_name,
        })

    cursor.close()
    conn.close()
    return ddl_entries


def get_bigquery_ddl(project_id: str, cred_file_path: str | None = None) -> list[dict]:
    """Extract table schemas from BigQuery via INFORMATION_SCHEMA."""
    from google.cloud import bigquery
    from google.oauth2 import service_account
    import json

    if cred_file_path:
        with open(cred_file_path, "r") as f:
            credentials = service_account.Credentials.from_service_account_info(
                json.loads(f.read()),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        client = bigquery.Client(project=project_id, credentials=credentials)
    else:
        client = bigquery.Client(project=project_id)

    # List all datasets
    datasets = list(client.list_datasets())
    ddl_entries = []

    for dataset in datasets:
        dataset_id = dataset.dataset_id
        query = f"""
            SELECT table_name, column_name, data_type, is_nullable
            FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            ORDER BY table_name, ordinal_position
        """
        try:
            rows = list(client.query(query).result())
        except Exception as e:
            print(f"  Skipping {dataset_id}: {e}")
            continue

        # Group by table
        tables: dict[str, list] = {}
        for row in rows:
            tables.setdefault(row.table_name, []).append(row)

        for table_name, columns in tables.items():
            full_name = f"{project_id}.{dataset_id}.{table_name}"
            col_defs = []
            for col in columns:
                col_def = f"  {col.column_name} {col.data_type}"
                if col.is_nullable == "NO":
                    col_def += " NOT NULL"
                col_defs.append(col_def)

            ddl = f"CREATE TABLE `{full_name}` (\n" + ",\n".join(col_defs) + "\n);"
            ddl_entries.append({
                "content": f"DDL for BigQuery table {full_name}:\n{ddl}",
                "table": full_name,
            })

    return ddl_entries


async def train(
    postgres_only: bool = False,
    bigquery_only: bool = False,
    fresh: bool = False,
    schemas: list[str] | None = None,
):
    memory = ChromaAgentMemory(
        persist_directory="./chroma_data",
        collection_name="vanna_memory",
    )
    ctx = get_dummy_context(memory)

    if fresh:
        print("Clearing old training data...")
        deleted = await memory.clear_memories(context=ctx)
        print(f"  Cleared {deleted} entries")

    total = 0

    # PostgreSQL
    pg_host = os.getenv("POSTGRES_HOST")
    if pg_host and not bigquery_only:
        pg_schemas = schemas or [
            s.strip() for s in os.getenv("TRAIN_SCHEMAS", "bronze,silver,gold").split(",") if s.strip()
        ]
        print(f"Loading PostgreSQL schemas ({', '.join(pg_schemas)})...")
        pg_conn = (
            f"host={pg_host} "
            f"port={os.getenv('POSTGRES_PORT', '5432')} "
            f"dbname={os.getenv('POSTGRES_DATABASE')} "
            f"user={os.getenv('POSTGRES_USER')} "
            f"password={os.getenv('POSTGRES_PASSWORD')}"
        )
        entries = get_postgres_ddl(pg_conn, pg_schemas)
        for entry in entries:
            await memory.save_text_memory(content=entry["content"], context=ctx)
            print(f"  Saved: {entry['table']}")
        total += len(entries)
        print(f"  Loaded {len(entries)} PostgreSQL tables")

    # BigQuery
    bq_project = os.getenv("BIGQUERY_PROJECT_ID")
    if bq_project and not postgres_only:
        print("Loading BigQuery schemas...")
        bq_cred_file = os.getenv("BIGQUERY_CREDENTIALS_FILE")
        entries = get_bigquery_ddl(bq_project, bq_cred_file)
        for entry in entries:
            await memory.save_text_memory(content=entry["content"], context=ctx)
            print(f"  Saved: {entry['table']}")
        total += len(entries)
        print(f"  Loaded {len(entries)} BigQuery tables")

    # Domain notes: competition/table-naming knowledge no DDL can express on its own.
    print("Loading domain notes...")
    for note in DOMAIN_NOTES:
        await memory.save_text_memory(content=note, context=ctx)
    total += len(DOMAIN_NOTES)
    print(f"  Loaded {len(DOMAIN_NOTES)} domain notes")

    print(f"\nDone! Loaded {total} total entries into ChromaDB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vanna with database schemas")
    parser.add_argument("--database", help="PostgreSQL database name (overrides POSTGRES_DATABASE from .env)")
    parser.add_argument("--postgres-only", action="store_true", help="Load only PostgreSQL schemas")
    parser.add_argument("--bigquery-only", action="store_true", help="Load only BigQuery schemas")
    parser.add_argument("--fresh", action="store_true", help="Clear old data before loading")
    parser.add_argument("--schemas", help="Comma-separated PostgreSQL schemas to load (overrides TRAIN_SCHEMAS from .env)")
    args = parser.parse_args()

    if args.database:
        os.environ["POSTGRES_DATABASE"] = args.database

    schemas = [s.strip() for s in args.schemas.split(",") if s.strip()] if args.schemas else None

    asyncio.run(train(
        postgres_only=args.postgres_only,
        bigquery_only=args.bigquery_only,
        fresh=args.fresh,
        schemas=schemas,
    ))
