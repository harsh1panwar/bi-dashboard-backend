# BI Dashboard

An interactive data analytics web application that lets users upload CSV files and ask questions about their data using plain English.

The application converts natural-language questions into SQLite SQL queries, executes them against the uploaded dataset, and returns visualization-ready results, an AI-generated insight, and follow-up questions.

## How It Works

```text
CSV Upload
    |
    v
SQLite in-memory session
    |
    v
Natural-language question
    |
    v
LLM generates SQLite SELECT
    |
    v
SQL execution
    |
    +----> Chart data
    +----> Executed SQL
    +----> One-line insight
    +----> Suggested follow-up questions
```

## Example

**Uploaded dataset**

```text
cars.csv
```

**Question**

```text
Show cars by fuel type
```

**Application returns**

- Interactive bar / line / pie / scatter visualization
- SQL query that was actually executed
- Short AI-generated insight
- 4 suggested follow-up questions

## Features

- Upload arbitrary CSV datasets
- Automatic column/type detection
- Session-based dataset handling
- Natural-language to SQLite SQL generation
- Read-only SELECT query validation
- Automatic chart type selection
- AI-generated one-line insights
- Context-aware follow-up queries
- Schema inspection
- Interactive HTML/JavaScript dashboard

## Technology Stack

### Backend
- Python
- FastAPI
- SQLite
- Pydantic

### AI / Query Generation
- Groq API

### Frontend
- HTML
- JavaScript
- Chart visualization

## API Endpoints

### Upload dataset

`POST /api/upload`

Uploads a CSV file and creates a new analysis session.

### Query dataset

`POST /api/query`

Accepts a natural-language question and returns query results, chart type, executed SQL, and an insight.

### Follow-up query

`POST /api/followup`

Modifies the previous SQL query according to a follow-up question.

### Get schema

`GET /api/schema/{session_id}`

Returns the current session's column names and inferred SQLite types.

## Session Model

Each CSV upload creates an isolated in-memory SQLite database connection.

The uploaded dataset is loaded into a single table:

```text
data
```

A generated session ID is used to associate subsequent API requests with the uploaded dataset.

## Running Locally

### Prerequisites

- Python 3.10+
- A Groq API key

### 1. Clone the repository

```bash
git clone https://github.com/harsh1panwar/bi-dashboard-backend.git
cd bi-dashboard-backend
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file:

```text
GROQ_API_KEY=<your-api-key>
```

Never commit `.env` or API keys.

### 5. Start the API

```bash
uvicorn main:app --reload
```

The FastAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Project Files

```text
main.py            # FastAPI backend and query pipeline
dashboard.html     # Frontend dashboard
requirements.txt   # Python dependencies
```

## Current Scope

The backend currently focuses on one uploaded dataset per session and supports analytical read-only SQL queries. External database connections are not required because the uploaded CSV becomes the session's data source.
