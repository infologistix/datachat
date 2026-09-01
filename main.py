"""
Self-hosted Vanna 2.0 server with Gemini LLM.

Connects to PostgreSQL and BigQuery databases.
Run with: python main.py
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from vanna.core.agent import Agent, AgentConfig
from vanna.core.registry import ToolRegistry
from vanna.core.user import User
from vanna.core.user.resolver import UserResolver
from vanna.core.user.request_context import RequestContext
from vanna.integrations.google.gemini import GeminiLlmService
from vanna.integrations.ollama.llm import OllamaLlmService
from vanna.integrations.openai.llm import OpenAILlmService
from vanna.integrations.chromadb.agent_memory import ChromaAgentMemory
from vanna.integrations.postgres.sql_runner import PostgresRunner
from vanna.integrations.bigquery.sql_runner import BigQueryRunner
from vanna.tools.run_sql import RunSqlTool
from vanna.tools.agent_memory import SearchSavedCorrectToolUsesTool, SaveQuestionToolArgsTool
from vanna.tools.visualize_data import VisualizeDataTool
from vanna.core.system_prompt import DefaultSystemPromptBuilder
from vanna.servers.base import ChatHandler
from vanna.servers.fastapi.routes import register_chat_routes

load_dotenv()


class LocalUserResolver(UserResolver):
    """Simple user resolver for personal/team use — returns a static admin user."""

    async def resolve_user(self, request_context: RequestContext) -> User:
        return User(
            id="local-user",
            username="admin",
            email="admin@localhost",
            group_memberships=["admin"],
        )


def create_app() -> FastAPI:
    app = FastAPI(title="Vanna Text-to-SQL")

    # LLM
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "ollama":
        llm = OllamaLlmService(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    elif provider == "openai":
        # Also used for any OpenAI-compatible endpoint (e.g. LiteLLM proxy in front of vLLM).
        llm = OpenAILlmService(
            model=os.getenv("OPENAI_MODEL", "gpt-5"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    else:
        llm = GeminiLlmService(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

    # Agent memory (ChromaDB)
    memory = ChromaAgentMemory(
        persist_directory="./chroma_data",
        collection_name="vanna_memory",
    )

    # Tools
    tools = ToolRegistry()

    # PostgreSQL
    pg_host = os.getenv("POSTGRES_HOST")
    if pg_host:
        pg_runner = PostgresRunner(
            host=pg_host,
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DATABASE"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )
        tools.register_local_tool(
            RunSqlTool(sql_runner=pg_runner),
            access_groups=[],
        )

    # BigQuery
    bq_project = os.getenv("BIGQUERY_PROJECT_ID")
    if bq_project:
        bq_cred_file = os.getenv("BIGQUERY_CREDENTIALS_FILE")
        bq_runner = BigQueryRunner(
            project_id=bq_project,
            cred_file_path=bq_cred_file,
        )
        tools.register_local_tool(
            RunSqlTool(
                sql_runner=bq_runner,
                custom_tool_name="run_bigquery_sql",
                custom_tool_description="Execute SQL queries against BigQuery",
            ),
            access_groups=[],
        )

    tools.register_local_tool(SearchSavedCorrectToolUsesTool(), access_groups=[])
    tools.register_local_tool(SaveQuestionToolArgsTool(), access_groups=[])
    
    # Visualization
    tools.register_local_tool(VisualizeDataTool(), access_groups=[])

    # System prompt — tell the agent what database it's connected to
    db_name = os.getenv("POSTGRES_DATABASE", "unknown")

    class CombinedSystemPromptBuilder(DefaultSystemPromptBuilder):
        def __init__(self, custom_context: str):
            super().__init__(base_prompt=None)  
            self.custom_context = custom_context

        async def build_system_prompt(self, user, tools):
            generated = await super().build_system_prompt(user, tools) 
            return (generated or "") + "\n\n" + self.custom_context 
    
    system_prompt_builder = CombinedSystemPromptBuilder(custom_context=f"""You are Zebrix, an AI basketball analyst assistant. Today's date is {__import__('datetime').date.today()}. # type: ignore # pyright: ignore[reportCallIssue]

    DATABASE: You are connected to a PostgreSQL database named '{db_name}'.
    - Use PostgreSQL syntax for all SQL queries.
    - To describe a table, use: SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = '<table>' ORDER BY ordinal_position
    - Do NOT use PRAGMA, DESCRIBE, or SHOW commands — those are for other databases.
    - Always use LIMIT instead of TOP for row limits.

    TABLE NAMING CONVENTION:
    - Tables use a layer prefix + league prefix, e.g. 'b_el_playbyplay'.
    - Layer prefixes: 'b_' = Bronze (raw), 's_' = Silver (cleansed), 'g_' = Gold (aggregated for BI).
    - League prefixes: 'el_' = Euroleague, 'bbl_' = Bundesliga, 'ec_' = Eurocup, 'cl_' = Championsleague.
    - Gold-layer tables without a league prefix span multiple leagues.
    - Tables with the same name after the league prefix hold the same kind of data across leagues.
    - If the user does not specify a league, default to Bundesliga (BBL).
    - If a season is not specified, default to the most recent season. Do not query across multiple seasons unless explicitly asked to.
    - A season runs roughly mid-July to mid-July (this dataset's season boundary is July 15) and is named for the two calendar years it spans.

    SEASON RESOLUTION:
    - CRITICAL: resolve relative season words against today's date above. "Last season" means the most recently COMPLETED season — this is NOT simply "today's year minus one". If today is after this season's July 15 boundary, "this season" is already the new one and "last season" is the one that just ended; if today is before the boundary, "this season" is still the one in progress. If a question names a season explicitly ("2024-25", "season 2024"), use that instead of any date math. 

    DATA GUIDELINES:
    - boxscore tables are usually best for individual/team game totals and rankings (one row per player per game). playbyplay tables are best for event sequences, shot/action timing, and possession-level questions (one row per event — can be 500+ rows per game).
    - CRITICAL: in boxscore AND playbyplay tables, the home/away team-total columns (e.g. home_team_total_point, away_team_total_point in EL/EC/CL — verify the exact column names for BBL, which uses date_final/home_team_final/away_team_final) are repeated identically on every row of a game. Summing these directly overcounts a team's points by roughly the roster size (boxscore) or event count (playbyplay, far worse), and plain COUNT(*) does not count games either. Before summing or counting games, always collapse to one row per team-game first (e.g. GROUP BY team + link, MAX() the total columns, or COUNT(DISTINCT link) for games played).
    - CRITICAL: when computing one team's win/loss outcome per game, check BOTH the case where that team was home AND where it was away — a condition that only tests the home side silently drops every away game, undercounting wins and losses.
    - Team names can have mid-season sponsor renames (e.g. old vs new name for the same club) that split aggregates unless normalized to one canonical name first. Normalize case-insensitively (ILIKE/UPPER) — team names are ALL CAPS in bronze tables but Title Case in gold tables, so a case-sensitive rename mapping silently fails to fire on gold.
    - Never filter a team or player name with exact equality when the name comes from how the user typed it — stored names are often ALL CAPS and won't match naturally-typed casing. Always use case-insensitive comparison (ILIKE or UPPER(column) = UPPER('literal')).

    Natural-language aliases:
    - points, score, total points, scored = pts
    - two_p, three_p = successfull two/three pointers, two_pa, three_pa = All two/three point throw attempts
    - player, athlete = player_name
    - team, club = team / home_team / away_team depending on table
    - minutes played = minutes
    - game, match = game_id or link depending on available columns
    - shot location, shot position = wurfposition tables

    VISUALIZATION:
    - run_sql only ever shows a raw results table in the UI — that is NOT a chart, graph,
    or diagram, regardless of how the data looks. Never describe a run_sql result table
    as "the chart" or "the graph" in your response.
    - If the user asks for a chart, graph, plot, trend, or diagram, you MUST call the
    visualize_data tool (using the filename returned by run_sql) as a separate step
    before you may claim any visualization was created. If you have not received a
    successful visualize_data tool result in this turn, do not say you created one.
    - The visualize_data tool has NO chart-type parameter — it infers the chart type purely
    from the shape of the CSV you feed it, not from what the user asked for:
    - exactly 1 datetime column + numeric column(s), 3 or fewer columns total -> line chart
    - exactly 1 categorical + 1 numeric column -> bar chart
    - exactly 2 numeric columns -> scatter plot
    - 1 numeric column only -> histogram
    - 4 or more columns -> ALWAYS a table, no matter what type was requested
    - 3+ numeric columns -> correlation heatmap
    - To produce the chart type the user actually asked for, shape your SQL SELECT to match
    the pattern above (e.g. for a requested line chart over time, select exactly a date/
    timestamp column plus the numeric metric(s), with no extra columns) BEFORE calling
    visualize_data.
    - The visualize_data tool result tells you the actual chart type it rendered. Always
    report that exact type back to the user — never assume it matches what they
    originally asked for, since the heuristic above may have picked something different.

    Response Guidelines:
    - When you execute a query, the raw result is shown to the user in the UI, so you do NOT need to repeat it. Focus on summarizing and interpreting.
    - Any summary or observations should be the final step.
    - Use the available tools to help the user accomplish their goals.
    - If you encounter a name containing initials do NOT guess what the name could be.
    - Ask for clarification instead of making up details or needlessly guess.
    - SQL should be a single SELECT statement; no DDL/DML. Add a LIMIT for broad row-list requests (100 or fewer).
    """)

    # Agent
    # Tool-call budget per question. The library default is 10, which questions
    # spanning several tables can exhaust before reaching an answer - the agent
    # then stops with "Tool Execution Limit Reached" instead of responding.
    # Configurable so it can be tuned without a rebuild.
    max_tool_iterations = int(os.getenv("MAX_TOOL_ITERATIONS", "20"))

    agent = Agent(
        llm_service=llm,
        tool_registry=tools,
        user_resolver=LocalUserResolver(),
        agent_memory=memory,
        system_prompt_builder=system_prompt_builder,
        config=AgentConfig(max_tool_iterations=max_tool_iterations),
    )

    # Schema explorer endpoint
    @app.get("/api/schema")
    async def get_schema():
        """Return database schema tree for the sidebar explorer."""
        import psycopg2
        import psycopg2.extras

        pg_host = os.getenv("POSTGRES_HOST")
        if not pg_host:
            return {"schemas": []}

        conn = psycopg2.connect(
            host=pg_host,
            port=os.getenv("POSTGRES_PORT", "5432"),
            dbname=os.getenv("POSTGRES_DATABASE"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT table_schema, table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name, ordinal_position
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Build tree: schema > table > columns
        schemas: dict = {}
        for row in rows:
            s = schemas.setdefault(row["table_schema"], {})
            t = s.setdefault(row["table_name"], [])
            t.append({
                "name": row["column_name"],
                "type": row["data_type"],
                "nullable": row["is_nullable"] == "YES",
            })

        result = []
        for schema_name, tables in sorted(schemas.items()):
            result.append({
                "name": schema_name,
                "tables": [
                    {"name": tname, "columns": cols}
                    for tname, cols in sorted(tables.items())
                ],
            })

        return {"schemas": result}

    # Serve local web component build
    static_dir = os.path.join(os.path.dirname(__file__), "frontends", "webcomponent", "dist")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Chat handler + routes (use local JS instead of CDN)
    chat_handler = ChatHandler(agent=agent)
    register_chat_routes(app, chat_handler, config={"cdn_url": "/static/vanna-components.js"})

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8084"))
    print(f"Starting Vanna server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
