import csv
import io
import os
import re
import sqlite3
import uuid

from dotenv import load_dotenv
load_dotenv()
from typing import Any
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

TABLE_NAME = "data"
sessions: dict[str, dict[str, Any]] = {}

app = FastAPI(title="BI Dashboard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_check():
    if GROQ_API_KEY:
        print(f"✅ GROQ_API_KEY loaded — starts with: {GROQ_API_KEY[:8]}...")
    else:
        print("❌ GROQ_API_KEY not found")

def get_session(session_id: str) -> dict[str, Any]:
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Session not found")
    return sessions[session_id]

def infer_sql_type(values: list[str]) -> str:
    if not values:
        return "TEXT"
    sample = [v.strip() for v in values if v is not None and str(v).strip() != ""]
    if not sample:
        return "TEXT"
    all_int = True
    all_float = True
    for v in sample[:100]:
        try:
            int(v)
        except (ValueError, TypeError):
            all_int = False
        try:
            float(v)
        except (ValueError, TypeError):
            all_float = False
    if all_int:
        return "INTEGER"
    if all_float:
        return "REAL"
    return "TEXT"

def get_schema_from_conn(conn: sqlite3.Connection) -> list[dict[str, str]]:
    cur = conn.execute(f"PRAGMA table_info({TABLE_NAME})")
    return [{"name": row[1], "type": row[2]} for row in cur.fetchall()]

def normalize_sql(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql.strip().upper()

def sql_starts_with_select(sql: str) -> bool:
    return normalize_sql(sql).startswith("SELECT")

def extract_sql_from_response(text: str) -> str | None:
    text = text.strip()
    match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines:
        if normalize_sql(line).startswith("SELECT"):
            return line
    if normalize_sql(text).startswith("SELECT"):
        return text
    return None

def run_query(conn: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    cur = conn.execute(sql)
    rows = cur.fetchall()
    names = [d[0] for d in cur.description] if cur.description else []
    return [dict(zip(names, row)) for row in rows]

def call_llm(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def _get_sample_values(session: dict[str, Any]) -> str:
    conn = session["conn"]
    schema = get_schema_from_conn(conn)
    samples = []
    for col in schema:
        if col["type"] == "TEXT":
            try:
                cur = conn.execute(
                    f'SELECT DISTINCT TRIM("{col["name"]}") FROM "{TABLE_NAME}" '
                    f'WHERE "{col["name"]}" IS NOT NULL LIMIT 5'
                )
                vals = [str(r[0]) for r in cur.fetchall()]
                samples.append(f'- {col["name"]} sample values: {", ".join(vals)}')
            except Exception:
                pass
    return "\n".join(samples) if samples else ""

def _build_sql_prompt(user_prompt: str, session: dict[str, Any]) -> str:
    sample_values = _get_sample_values(session)
    columns_list = "\n".join([f'- "{c}"' for c in session["columns"]])
    return (
        f"You are a SQLite expert. The table is called `data`.\n\n"
        f"EXACT column names (use these exactly with double quotes):\n"
        f"{columns_list}\n\n"
        f"Sample text values:\n{sample_values}\n\n"
        f"STRICT rules — follow ALL:\n"
        f"1. Wrap EVERY column name in double quotes: \"fuelType\", \"model\", \"price\"\n"
        f"2. Always TRIM() text columns: TRIM(\"model\")\n"
        f"3. Always use GROUP BY with aggregate functions (COUNT, AVG, SUM)\n"
        f"4. Always ROUND decimal results: ROUND(AVG(\"price\"), 0)\n"
        f"5. Always ORDER BY the numeric column DESC\n"
        f"6. Always end with LIMIT 20\n"
        f"7. NEVER use LIMIT 1\n"
        f"8. NEVER use SELECT *\n"
        f"9. Always return exactly 2 columns: one label + one value\n"
        f"10. COUNT example: SELECT TRIM(\"fuelType\") as fuel_type, COUNT(*) as total_cars FROM data GROUP BY TRIM(\"fuelType\") ORDER BY total_cars DESC LIMIT 20\n\n"
        f"Write ONE SQLite SELECT query for: {user_prompt}\n"
        f"Return ONLY the SQL. No explanation. No markdown backticks."
    )

def _build_followup_sql_prompt(followup_prompt: str, previous_query: str, session: dict[str, Any]) -> str:
    sample_values = _get_sample_values(session)
    columns_list = "\n".join([f'- "{c}"' for c in session["columns"]])
    return (
        f"You are a SQLite expert. The table is called `data`.\n\n"
        f"EXACT column names (use with double quotes):\n"
        f"{columns_list}\n\n"
        f"Sample text values:\n{sample_values}\n\n"
        f"STRICT rules:\n"
        f"1. Wrap EVERY column name in double quotes\n"
        f"2. Always TRIM() text columns\n"
        f"3. Always GROUP BY with aggregate functions\n"
        f"4. Always ROUND decimal results\n"
        f"5. Always ORDER BY value DESC\n"
        f"6. Always LIMIT 20 — NEVER LIMIT 1\n"
        f"7. Always return exactly 2 columns: one label + one value\n\n"
        f"Previous SQL query:\n{previous_query}\n\n"
        f"Modify it based on: {followup_prompt}\n"
        f"Return ONLY the new SQL. No explanation. No markdown backticks."
    )

class QueryRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from upload")
    prompt: str     = Field(..., description="Natural language query")

class FollowupRequest(BaseModel):
    session_id:      str = Field(..., description="Session ID from upload")
    previous_query:  str = Field(..., description="Previously run SQL")
    followup_prompt: str = Field(..., description="Follow-up instruction")

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded")
    if text.lstrip().startswith("bplist00"):
        raise HTTPException(status_code=400, detail="File does not appear to be a valid CSV.")
    reader = csv.reader(io.StringIO(text))
    try:
        rows = list(reader)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {e}")
    if not rows:
        raise HTTPException(status_code=400, detail="CSV is empty")

    headers   = [h.strip() or f"col_{i}" for i, h in enumerate(rows[0])]
    data_rows = rows[1:]

    col_types = []
    for col_idx in range(len(headers)):
        values = [row[col_idx] if col_idx < len(row) else "" for row in data_rows[:500]]
        col_types.append(infer_sql_type(values))

    safe_headers = []
    for h in headers:
        safe = re.sub(r"[^\w]", "_", h) or "col"
        if safe[0].isdigit():
            safe = "c_" + safe
        safe_headers.append(safe)

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    placeholders = ", ".join(["?" for _ in safe_headers])
    columns_ddl  = ", ".join(f'"{c}" {t}' for c, t in zip(safe_headers, col_types))
    conn.execute(f'CREATE TABLE "{TABLE_NAME}" ({columns_ddl})')

    for row in data_rows:
        padded = (row[i].strip() if i < len(row) else "" for i in range(len(safe_headers)))
        conn.execute(f'INSERT INTO "{TABLE_NAME}" VALUES ({placeholders})', list(padded))

    conn.commit()
    session_id      = str(uuid.uuid4())
    columns_from_db = [r[1] for r in conn.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()]
    sessions[session_id] = {"conn": conn, "columns": columns_from_db}
    return {"session_id": session_id, "columns": columns_from_db}

@app.get("/api/schema/{session_id}")
async def get_schema(session_id: str):
    session = get_session(session_id)
    schema  = get_schema_from_conn(session["conn"])
    return {
        "columns": [s["name"] for s in schema],
        "types":   {s["name"]: s["type"] for s in schema},
    }

@app.post("/api/query")
async def query(request: QueryRequest):
    if groq_client is None:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set")

    session    = get_session(request.session_id)
    sql_prompt = _build_sql_prompt(request.prompt, session)

    try:
        sql_raw = call_llm(sql_prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {str(e)}")

    sql = extract_sql_from_response(sql_raw)
    if not sql or not sql_starts_with_select(sql):
        raise HTTPException(status_code=400, detail="Invalid SELECT query generated")

    try:
        data = run_query(session["conn"], sql)
    except sqlite3.Error as e:
        raise HTTPException(status_code=400, detail=f"SQL execution failed: {str(e)}")

    if not data:
        raise HTTPException(status_code=400, detail="Query returned no results. Try rephrasing.")

    chart_prompt = (
        f"User asked: {request.prompt}\n"
        f"Data has {len(data)} rows with columns: {list(data[0].keys())}\n"
        f"Rules: 'line' for year/time trends, 'pie' for parts-of-whole with fewer than 8 items, "
        f"'bar' for comparisons, 'scatter' for correlations.\n"
        f"Reply with ONE word only: bar, line, pie, or scatter"
    )
    try:
        chart_type = call_llm(chart_prompt).strip().lower()
        for ct in ("bar", "line", "pie", "scatter"):
            if ct in chart_type:
                chart_type = ct
                break
        else:
            chart_type = "bar"
    except Exception:
        chart_type = "bar"

    return {"data": data, "chart_type": chart_type, "query_used": sql}

@app.post("/api/followup")
async def followup(request: FollowupRequest):
    if groq_client is None:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set")

    session    = get_session(request.session_id)
    sql_prompt = _build_followup_sql_prompt(
        request.followup_prompt, request.previous_query, session
    )

    try:
        sql_raw = call_llm(sql_prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {str(e)}")

    sql = extract_sql_from_response(sql_raw)
    if not sql or not sql_starts_with_select(sql):
        raise HTTPException(status_code=400, detail="Invalid SELECT query generated")

    try:
        data = run_query(session["conn"], sql)
    except sqlite3.Error as e:
        raise HTTPException(status_code=400, detail=f"SQL execution failed: {str(e)}")

    if not data:
        raise HTTPException(status_code=400, detail="Query returned no results. Try rephrasing.")

    chart_prompt = (
        f"User asked: {request.followup_prompt}\n"
        f"Data has {len(data)} rows with columns: {list(data[0].keys())}\n"
        f"Rules: 'line' for year/time trends, 'pie' for parts-of-whole with fewer than 8 items, "
        f"'bar' for comparisons, 'scatter' for correlations.\n"
        f"Reply with ONE word only: bar, line, pie, or scatter"
    )
    try:
        chart_type = call_llm(chart_prompt).strip().lower()
        for ct in ("bar", "line", "pie", "scatter"):
            if ct in chart_type:
                chart_type = ct
                break
        else:
            chart_type = "bar"
    except Exception:
        chart_type = "bar"

    return {"data": data, "chart_type": chart_type, "query_used": sql}

@app.get("/")
async def root():
    return {"message": "BI Dashboard API is running", "docs": "/docs"}