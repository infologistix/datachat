"""
Training script for Vanna — loads table DDL into ChromaDB.

Everything loaded here is read from the database itself (information_schema):
table names, columns, types, nullability, defaults, and primary keys. No
hand-written domain knowledge is injected.

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

        # Primary key, when one is declared. Only a minority of tables have one:
        # dbt's materialized='table' does CREATE TABLE AS SELECT on every run,
        # which drops constraints, so most of silver/gold has none. Where a PK
        # does exist it states the table's grain (e.g. boxscore is keyed on
        # (date, home_team, player_name) - one row per player per game), which
        # is exactly the kind of thing a column list alone does not convey.
        cursor.execute("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.constraint_schema = kcu.constraint_schema
             AND tc.table_name = kcu.table_name
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema = %s AND tc.table_name = %s
            ORDER BY kcu.ordinal_position
        """, (schema, name))
        pk_cols = [r["column_name"] for r in cursor.fetchall()]

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

        if pk_cols:
            col_defs.append(f"  PRIMARY KEY ({', '.join(pk_cols)})")

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

    # You can add custom documentation and example SQL here:
    memories = [
        # rebound-rate scale confusion + net_rating_off_court
        "gold.g_bbl_players has ofreb_rate/defreb_rate: a PLAYER-level individual "
        "rebound rate on a 0-100 PERCENT scale (e.g. 23.8 means 23.8%). This is a "
        "different metric from of_per_eigenes/def_per_eigenes in gold.g_*_teams, "
        "which is a TEAM-level rebound rate on a 0-1 FRACTION scale (e.g. 0.335). "
        "They look similar by name but are different grains and different scales — "
        "do not use one where the question asks for the other, and do not compare "
        "them directly without converting scale first.",
        #
        "net_rating_off_court (gold.g_bbl_players and equivalents in other leagues) "
        "is the TEAM's net rating during the minutes this specific player was NOT on "
        "the floor — an on/off-style stat, not the player's own offensive rating and "
        "not 'rating while playing offense'. A question asking for a player's own "
        "offensive quality should use offensive_rating instead. A high "
        "net_rating_off_court can mean the team does fine without that player, closer "
        "to 'replaceable' than 'outstanding' — the opposite of what the name suggests. "
        "There is no net_rating_on_court column; an on/off difference for a player "
        "would need to be derived from the team's overall net rating in "
        "gold.g_*_teams minus this column.",

        # --- position value encoding (differs by league) ---
        "Never guess a position value (e.g. 'C', 'PG') — always match against the exact "
        "stored string, which differs by league. EL/EC use full words only ('Guard', "
        "'Forward', 'Center'), no abbreviations exist. CL mixes two granularities in the "
        "same column: ('Guard', 'Forward', 'Center') AND ('Point Guard', 'Shooting Guard', "
        "'Small Forward', 'Power Forward') — a query for 'guards' should match both forms "
        r"(position ILIKE '%guard%'), not just one. BBL is the messiest: full-name and "
        "abbreviation forms coexist, sometimes combined in one string like "
        "'Power Forward (PF)' and sometimes bare 'PF' or 'C' — always match BBL positions "
        "with ILIKE '%pattern%', never exact equality. Positions are not directly "
        "comparable across leagues in the same query because of these differing "
        "conventions.",

        # --- nationality encoding (differs by league) ---
        "The nationality column is encoded differently in every league — a country filter "
        "that works for one league can silently return zero rows for another. "
        "b_bbl_player_info uses German country names with a 2-letter ISO code in brackets "
        "('Deutschland (DE)'), and inconsistently — some rows omit the bracket. "
        "b_el_player_info and b_ec_player_info use English full names (note EC writes "
        "'Turkiye', not 'Turkey'). b_cl_player_info uses 3-letter uppercase codes ('GER', "
        "'USA', 'FRA'). Never compare nationality with '=' — always ILIKE '%...%', because "
        "equality misses both the bracket-less BBL variant and dual nationals. Dual "
        "nationality is stored in a single cell, separated by a newline in BBL or a comma "
        "in CL — a substring ILIKE handles both, but equality or GROUP BY on the raw value "
        "treats each combination as its own country. There is no normalized country column "
        "— a question spanning several leagues needs a separate filter per league.",

        # --- BBL player_name join bridge ---
        "For BBL specifically, b_bbl_boxscore.player_name (e.g. 'Mikesell R. (SF)' — "
        "abbreviated first name, trailing '(POS)' suffix, mixed case) does NOT match "
        "b_bbl_player_info.player_name (e.g. 'RYAN MIKESELL' — full name, all caps, no "
        "suffix) — joining on player_name directly returns zero rows. The bridge column is "
        "b_bbl_player_info.player_name_short (e.g. 'MIKESELL R.' — all caps, no suffix): "
        "strip the '(POS)' suffix from boxscore's player_name and uppercase it before "
        "joining, e.g. UPPER(regexp_replace(boxscore.player_name, '\\s*\\(.*\\)\\s*$', '')) "
        "= player_info.player_name_short. Also note b_bbl_player_info can carry multiple, "
        "sometimes conflicting, position strings for the same player across different "
        "crawled records — use ILIKE matching rather than expecting one canonical value "
        "per player.",        

         # --- player_info missing season scoping ---
        "*_player_info tables (position, nationality, height, birthday) have NO season or "
        "date column — they are a rolling multi-season/career player registry, not one "
        "season's roster, and the same player can appear multiple times across different "
        "crawled seasons or team changes. A question that names a season but is answered "
        "purely from *_player_info silently mixes every season the crawler has ever seen. "
        "Whenever a question about position/nationality/height/etc. also names a season, "
        "first find that season's actual player set from a season-dated source (boxscore "
        "filtered by date, or a gold table with saison) and only then join to player_info "
        "for the attribute.",

        # --- chronological sort direction ---
        "Get the sort direction right for 'youngest'/'oldest' and similar chronological "
        "superlatives — a birthday column sorted ASC gives the earliest date, which is the "
        "OLDEST person, not the youngest. 'Youngest' needs ORDER BY birthday DESC (most "
        "recent birthdate first); 'oldest' needs ORDER BY birthday ASC. Double-check this "
        "inversion risk on any 'most recent'/'earliest'/'newest'/'latest' question "
        "involving a date column too, not just birthday.",

        # --- ef vs eFG% ---
        "ef (boxscore tables) is the classic basketball 'Efficiency' (EFF) rating — "
        "(PTS+REB+AST+STL+BLK) minus (missed FG + missed FT + TOV) — an unbounded counting "
        "number (can run 40+), NOT effective field goal percentage (eFG%, a 0-100% "
        "shooting-efficiency stat computed from FGM/three-pointers/FGA). A question asking "
        "for 'eFG%' or 'effective field goal percentage' needs FGM/FGA/three_p computed "
        "directly — the ef column is an unrelated statistic that happens to share a "
        "similar-looking abbreviation.",

        # --- tm-prefixed / opp-prefixed on-court-subset columns ---
        "The tm-prefixed columns in silver.*_player_stats_prospiel (tmpts, tmast, "
        "tmofreb, tmdefreb, tmstl, tmto) are the TEAM's totals accumulated only during "
        "this player's on-court minutes for that game — not the player's own individual "
        "stat (that's the bare pts/ast/etc. in boxscore tables) and not the team's "
        "full-game total. Do not sum tmpts across a player's games to get 'team points' — "
        "it double-counts every possession the team's other rotation players were also on "
        "court for. The opp-prefixed columns (opppts, oppast, oppdefreb, oppofreb, oppto) "
        "are the OPPONENT's totals at the same on-court-subset grain, not the opponent's "
        "full-game total either.",

        # --- _rate column scale inconsistency ---
        "Within gold.g_bbl_players, columns ending in _rate do not share one numeric "
        "scale. Some (e.g. tmast_rate, tmtov_rate_eigenes) are 0-1 fractions, others (e.g. "
        "tm_twop_rate, tm_threep_rate) are 0-100 percentages, and others still (e.g. "
        "tmstl_rate, tm_ft_trip_rate) are a per-100-possessions-style rate on neither "
        "scale. Never assume a common scale across different _rate columns, even within "
        "the same table or the same tm-prefixed family — check a sample value's plausible "
        "range (0-1, 0-100, or per-100) before using one in a calculation, comparison, or "
        "threshold.",

        # --- wurfposition coordinate column naming ---
        "bronze.b_bbl_wurfposition uses xcoordinate/ycoordinate (no underscore) while "
        "bronze.b_el_wurfposition and silver.b_el_wurfposition use x_coordinate/"
        "y_coordinate (with underscore) for the same concept. Using the wrong league's "
        "spelling raises a column-does-not-exist error — check which wurfposition table "
        "is in play before writing the column name from memory.",

        # --- BBL phantom cup opponents ---
        "BBL bronze/silver team rows include ProA cup opponents that are not real "
        "Bundesliga teams — they appear only 1-2 times all season (vs. 33+ for a real BBL "
        "team), with no clean structural filter (no competition/spieltyp column exists). "
        "Any BBL team-level aggregation ('which team scored the most', 'how many teams', "
        "bottom-N rankings, averages across teams) must exclude these with HAVING "
        "COUNT(*) > 2 on the team-game grouping, or it silently counts phantom cup "
        "opponents as BBL teams.",

        # --- average-vs-leaderboard bias + season-total qualifier ---
        "A question asking for the average/mean/typical value of a stat across players or "
        "teams (not asking to rank or find the top/best) must be answered with a single "
        "aggregate computed over ALL qualifying rows — e.g. AVG(pct) or a weighted "
        "SUM(made)*100.0/SUM(attempts) — never with a ranked leaderboard (ORDER BY ... "
        "LIMIT N) whose top-N rows are then averaged, which silently drops everyone "
        "outside the top N and biases the result high. When such an average also has a "
        "'minimum N attempts/games' qualifier, N is a SEASON TOTAL per player, never a "
        "per-row/per-game threshold — use a two-level aggregate: first GROUP BY player "
        "with HAVING SUM(x) >= N to get qualifying player-seasons, then aggregate that "
        "qualifying set into the final number.",
    ]

    for note in memories:
        await memory.save_text_memory(content=note, context=ctx)
        total+=1
    print(f' Saved {len(memories)} new memories ')

    # Example:
    # await memory.save_text_memory(
    #     content="The 'orders' table contains all customer orders. "
    #             "Use order_date for time-based filtering.",
    #     context=ctx,
    # )

    examples = [
        {
            "question": "Welche Bundesliga Mannschaft hat die meisten Rebounds in der Saison 2025-2026",
            "sql": """\
                WITH teams AS (
                    SELECT
                        home_team_final AS team,
                        orb,
                        drb
                    FROM bronze.b_bbl_boxscore
                    WHERE date_final >= '2025-07-15'
                    AND date_final < '2026-07-15'
                    UNION ALL
                    SELECT
                        away_team_final AS team,
                        orb,
                        drb
                    FROM bronze.b_bbl_boxscore
                    WHERE date_final >= '2025-07-15'
                    AND date_final < '2026-07-15'
                ),
                team_sum_reb AS (
                    SELECT
                        team,
                        SUM(orb) AS orb_ges,
                        SUM(drb) AS drb_ges
                    FROM teams
                    GROUP BY team
                )
                SELECT
                    team,
                    REB_GES
                FROM (
                    SELECT
                        team,
                        orb_ges + drb_ges AS REB_GES,
                        ROW_NUMBER() OVER (ORDER BY orb_ges DESC) AS rn
                    FROM team_sum_reb
                ) t
                WHERE rn <= 5
                ORDER BY rn;""".strip()
        },
        {
            "question": "Welcher Spieler in der Bundesliga hat die meisten Punkte erzielt in der Saison 2025-2026?",
            "sql": """\
                SELECT
                    player_name,
                    player_team,
                    pts_ges
                FROM (
                    SELECT
                        player_name,
                        player_team,
                        SUM(pts) AS pts_ges,
                        RANK() OVER (ORDER BY SUM(pts) DESC) AS rnk
                    FROM bronze.b_bbl_boxscore
                    WHERE date_final >= '2025-07-15'
                    AND date_final < '2026-07-15'
                    GROUP BY player_name, player_team
                ) t
                WHERE rnk = 1;""".strip()
        }
    ]

    for ex in examples:
        await memory.save_tool_usage(
            question=ex['question'],
            tool_name='run_sql',
            args={'sql': ex['sql']},
            context=ctx,
            success=True
        )
        print(f" Saved example {ex['question']}")
    total += len(examples)

    # await memory.save_tool_usage(
    #     question="How many orders were placed last month?",
    #     tool_name="run_sql",
    #     args={"sql": "SELECT COUNT(*) FROM orders WHERE order_date >= NOW() - INTERVAL '1 month'"},
    #     context=ctx,
    # )

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
