import sqlalchemy
from sqlalchemy import create_engine, text, inspect
import hashlib
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import logging
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import tiktoken
from langchain_community.chat_message_histories import ChatMessageHistory
from openai import OpenAI
import urllib.parse
import sqlite3
import jwt
from datetime import datetime, timedelta
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import decimal  # Added import
from datetime import date
from typing import Tuple

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")
JWT_ALGORITHM = "HS256"

# Validate environment variables
if not OPENROUTER_API_KEY:
    logger.error("Missing required environment variable: OPENROUTER_API_KEY must be set")
    raise ValueError("Missing required environment variable")

# Initialize OpenAI client with OpenRouter
llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Default schema for fallback (used only if bootstrapping fails)
DEFAULT_SCHEMA = {
    "hospitals": {
        "columns": [
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "name", "type": "VARCHAR(255)", "primary_key": False}
        ],
        "primary_key": ["id"],
        "foreign_keys": []
    },
    "patients": {
        "columns": [
            {"name": "id", "type": "INTEGER", "primary_key": True}
        ],
        "primary_key": ["id"],
        "foreign_keys": []
    },
    "cases": {
        "columns": [
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "hospital_id", "type": "INTEGER", "primary_key": False},
            {"name": "patient_id", "type": "INTEGER", "primary_key": False}
        ],
        "primary_key": ["id"],
        "foreign_keys": [
            {"column": ["hospital_id"], "referred_table": "hospitals", "referred_columns": ["id"]},
            {"column": ["patient_id"], "referred_table": "patients", "referred_columns": ["id"]}
        ]
    }
}

def create_access_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode = {"sub": str(user_id), "exp": expire}
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    logger.info(f"Generated JWT token for user_id {user_id}")
    return encoded_jwt

def get_user_id_from_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Optional[int]:
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = int(payload.get("sub"))
        logger.info(f"Extracted user_id from token: {user_id}")
        return user_id
    except jwt.PyJWTError as e:
        logger.warning(f"Invalid or missing token: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

user_memories = {}

class UserCredentials(BaseModel):
    username: str
    password: str

class DatabaseCredentials(BaseModel):
    db_host: str
    db_port: str
    db_user: str
    db_password: str
    db_name: str

class BootstrapSchemaRequest(BaseModel):
    db_credentials: DatabaseCredentials

class QueryRequest(BaseModel):
    db_credentials: DatabaseCredentials
    schema_text: Optional[str] = None
    query: str
    user_id: Optional[int] = None
    conversation: Optional[List[Dict[str, str]]] = []
    export_format: str = "None"
    use_enhanced: bool = False
    enhanced_prompt: Optional[str] = None
    chart_type: str = "none"

class EnhancePromptRequest(BaseModel):
    db_credentials: DatabaseCredentials
    query: str
    schema_text: Optional[str] = None

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    try:
        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                prompt TEXT NOT NULL,
                sql_query TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            )
        """)
        conn.commit()
        logger.info("SQLite user database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SQLite database: {str(e)}")
        raise
    finally:
        conn.close()

init_db()

@app.post("/api/login")
async def login(credentials: UserCredentials):
    try:
        user = login_user(credentials.username, credentials.password)
        if not user:
            logger.warning(f"Login failed for username: {credentials.username}")
            raise HTTPException(status_code=401, detail="Invalid username or password")
        return {
            "user_id": user["user_id"],
            "access_token": user["access_token"]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

def add_user(username: str, password: str):
    try:
        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    except Exception as e:
        logger.error(f"Error adding user: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

def login_user(username: str, password: str):
    try:
        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username=? AND password=?", (username, hash_password(password)))
        user = cursor.fetchone()
        if user:
            return {"user_id": user[0], "access_token": create_access_token(user[0])}
        return None
    except Exception as e:
        logger.error(f"Error logging in user: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

def count_tokens(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(f"Model {model} not found in tiktoken, using cl100k_base")
        enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        total += len(enc.encode(msg.get("content", "")))
    return total

def trim_prompt(messages: List[Dict[str, str]], schema_text: str, max_allowed: int = 15000, model: str = "gpt-4o-mini") -> tuple[List[Dict[str, str]], str]:
    tokens = count_tokens(messages, model) + len(schema_text.split())
    while tokens > max_allowed and len(messages) > 3:
        messages.pop(1)
        tokens = count_tokens(messages, model) + len(schema_text.split())
    if tokens > max_allowed:
        schema_text = " ".join(schema_text.split()[:2000])
    return messages, schema_text

def save_query(user_id: Optional[int], prompt: str, sql_query: str = None):
    try:
        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_queries (user_id, prompt, sql_query) VALUES (?, ?, ?)",
            (user_id, prompt, sql_query)
        )
        conn.commit()
        logger.info(f"Query saved successfully for user_id: {user_id}")
    except Exception as e:
        logger.error(f"Error saving query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save query: {str(e)}")
    finally:
        conn.close()

def get_user_queries(user_id: Optional[int], limit: int = None):
    try:
        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()
        if user_id is None:
            query = """
                SELECT id, prompt, sql_query, created_at
                FROM user_queries
                WHERE user_id IS NULL
                ORDER BY created_at DESC
            """
            params = []
        else:
            query = """
                SELECT id, prompt, sql_query, created_at
                FROM user_queries
                WHERE user_id = ?
                ORDER BY created_at DESC
            """
            params = [user_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        logger.error(f"Error fetching user queries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch query history: {str(e)}")
    finally:
        conn.close()

def get_schema_file_path(credentials: DatabaseCredentials) -> str:
    cred_string = f"{credentials.db_host}:{credentials.db_port}:{credentials.db_user}:{credentials.db_name}"
    cred_hash = hashlib.md5(cred_string.encode()).hexdigest()
    return f"schema_{cred_hash}.txt"  # Changed extension to .txt

def bootstrap_schema(db_credentials: DatabaseCredentials) -> str:
    """Extract trimmed schema info (columns, datatypes, PK, FK) and save as delimited text file.

    Args:
        db_credentials (DatabaseCredentials): Database connection credentials.

    Returns:
        str: Delimited schema string.

    Raises:
        HTTPException: If database connection fails or schema extraction fails.
    """
    try:
        db_url = f"mysql+pymysql://{db_credentials.db_user}:{urllib.parse.quote_plus(db_credentials.db_password)}@{db_credentials.db_host}:{db_credentials.db_port}/{db_credentials.db_name}"
        logger.info(f"Attempting to connect to database: {db_credentials.db_name} at {db_credentials.db_host}:{db_credentials.db_port}")
        engine = create_engine(db_url)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
            # Check for tables
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            logger.info(f"Found tables: {table_names}")
            if not table_names:
                logger.error(f"No tables found in database {db_credentials.db_name}")
                raise HTTPException(status_code=400, detail=f"No tables found in database {db_credentials.db_name}. Please ensure the database contains tables like 'hospitals', 'cases', 'patients'.")

        schema_parts = []
        for table_name in table_names:
            # Columns
            columns_info = inspector.get_columns(table_name)
            trimmed_columns = [
                f"{col['name']}:{str(col['type']).split('(')[0] if '(' in str(col['type']) else str(col['type'])}"
                for col in columns_info
            ]
            columns_str = ",".join(trimmed_columns)

            # Primary Key
            pk = inspector.get_pk_constraint(table_name)
            trimmed_pk = pk.get("constrained_columns", [])[0] if pk.get("constrained_columns") else ""

            # Foreign Keys
            fks_info = inspector.get_foreign_keys(table_name)
            trimmed_fks = {}
            for fk in fks_info:
                if fk["constrained_columns"]:
                    trimmed_fks[fk["constrained_columns"][0]] = f"{fk['referred_table']}.{fk['referred_columns'][0]}"
            fks_str = ",".join(f"{k}->{v}" for k, v in trimmed_fks.items()) if trimmed_fks else ""

            # Construct delimited string
            table_str = f"{table_name}|{columns_str}|primary_key:{trimmed_pk}|foreign_keys:{fks_str}"
            schema_parts.append(table_str)

        schema_str = "\n".join(schema_parts)
        logger.info("Trimmed schema generated successfully")

        # Save trimmed schema to text file
        schema_file = get_schema_file_path(db_credentials)
        try:
            with open(schema_file, "w", encoding='utf-8') as f:
                f.write(schema_str)
            logger.info(f"Trimmed schema successfully saved to {schema_file}")
            logger.debug(f"Trimmed schema content: {schema_str}")
        except Exception as e:
            logger.error(f"Failed to save schema file {schema_file}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save schema file: {str(e)}")

        return schema_str

    except sqlalchemy.exc.OperationalError as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Database connection failed: {str(e)}. Please check your credentials and ensure the database is accessible.")
    except Exception as e:
        logger.error(f"Failed to bootstrap schema: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to bootstrap schema: {str(e)}")
    finally:
        if 'engine' in locals():
            engine.dispose()
            logger.info("Database engine disposed")

def load_schema(db_credentials: DatabaseCredentials) -> str:
    schema_file = get_schema_file_path(db_credentials)
    logger.info(f"Attempting to load schema from {schema_file}")
    if os.path.exists(schema_file):
        try:
            with open(schema_file, "r", encoding='utf-8') as f:
                schema_str = f.read().strip()
                if not schema_str:
                    logger.warning(f"Schema file {schema_file} is empty, re-bootstrapping schema")
                    schema_str = bootstrap_schema(db_credentials)
                else:
                    logger.info(f"Schema loaded successfully from {schema_file}")
                    logger.debug(f"Schema content: {schema_str}")
                return schema_str
        except Exception as e:
            logger.error(f"Failed to load schema from {schema_file}: {str(e)}")
            logger.info("Re-bootstrapping schema due to file access error")
            return bootstrap_schema(db_credentials)
    else:
        logger.warning(f"Schema file {schema_file} not found, bootstrapping new schema")
        try:
            return bootstrap_schema(db_credentials)
        except HTTPException as e:
            logger.error(f"Failed to bootstrap schema, using default schema: {str(e)}")
            default_schema_str = schema_to_text(DEFAULT_SCHEMA)
            logger.info(f"Using default schema: {default_schema_str}")
            return default_schema_str



def schema_to_text(schema: Dict, output_file: str = None, output_format: str = "text") -> str:
    if not schema:
        logger.error("Schema is empty in schema_to_text, using default schema")
        schema = DEFAULT_SCHEMA
    
    lines = []
    for table, details in schema.items():
        lines.append(f"Table: {table}")
        for col in details.get("columns", []):
            lines.append(f"  {col['name']} {col['type']}")
        if details.get("primary_key"):
            lines.append(f"  PRIMARY KEY: {', '.join(details['primary_key'])}")
        for fk in details.get("foreign_keys", []):
            lines.append(f"  FOREIGN KEY: {fk['column']} -> {fk['referred_table']}({fk['referred_columns']})")
    
    schema_text = "\n".join(lines)
    logger.debug(f"Schema text generated: {schema_text}")
    
    if output_file:
        try:
            if output_format == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({"schema_text": schema_text}, f, indent=2)
            else:  # Default to text
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(schema_text)
            logger.info(f"Schema text successfully saved to {output_file} as {output_format}")
        except Exception as e:
            logger.error(f"Failed to save schema text to {output_file}: {str(e)}")
            raise
        
    
    return schema_text

def filter_schema(schema: str, user_prompt: str, max_chars: int = 2000) -> str:
    if not schema:
        logger.error("Empty schema provided to filter_schema, using default schema")
        schema = schema_to_text(DEFAULT_SCHEMA)
    filtered_lines = []
    prompt_words = user_prompt.lower().split()
    core_tables = ['cases', 'patients', 'case_answers', 'questions', 'hospitals']
    for line in schema.splitlines():
        line_lower = line.lower()
        if any(word in line_lower for word in prompt_words) or any(table in line_lower for table in core_tables):
            filtered_lines.append(line)
    reduced = "\n".join(filtered_lines)
    if not reduced:
        lines = schema.splitlines()
        filtered_lines = [line for line in lines if any(table in line.lower() for table in core_tables)]
        reduced = "\n".join(filtered_lines) if filtered_lines else schema[:max_chars]
    logger.debug(f"Filtered schema: {reduced}")
    return reduced[:max_chars]

def extract_schema_info(schema: str) -> dict:
    if not schema:
        logger.error("Empty schema provided to extract_schema_info, using default schema")
        schema = schema_to_text(DEFAULT_SCHEMA)
    tables = {}
    current_table = None
    for line in schema.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('Table:'):
            current_table = line.replace('Table:', '').strip()
            tables[current_table] = []
        elif current_table and line.startswith('  ') and not line.startswith('  PRIMARY KEY') and not line.startswith('  FOREIGN KEY'):
            column_name = line.strip().split()[0].replace('`', '').replace(',', '')
            if column_name:
                tables[current_table].append(column_name)
    if not tables:
        logger.error("No tables extracted from schema, using default schema")
        schema_dict = DEFAULT_SCHEMA
        tables = {table: [col['name'] for col in details['columns']] for table, details in schema_dict.items()}
    logger.debug(f"Extracted schema info: {json.dumps(tables, indent=2)}")
    return tables

def init_mysql_request_limits(db_credentials: DatabaseCredentials):
    try:
        db_url = f"mysql+pymysql://{db_credentials.db_user}:{urllib.parse.quote_plus(db_credentials.db_password)}@{db_credentials.db_host}:{db_credentials.db_port}/{db_credentials.db_name}"
        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_request_limits (
                    user_id INTEGER,
                    request_date DATE,
                    request_count INTEGER DEFAULT 0,
                    PRIMARY KEY (user_id, request_date)
                )
            """))
        logger.info("MySQL user_request_limits table initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MySQL user_request_limits table: {str(e)}")
        raise

# Updated check_and_increment_request_limit
def check_and_increment_request_limit(user_id: int, db_credentials: DatabaseCredentials, limit: int = 3) -> Tuple[bool, int]:
    """
    Returns (allowed, remaining_requests).
    - allowed: whether the request is allowed
    - remaining_requests: how many left today
    """
    try:
        db_url = f"mysql+pymysql://{db_credentials.db_user}:{urllib.parse.quote_plus(db_credentials.db_password)}@{db_credentials.db_host}:{db_credentials.db_port}/{db_credentials.db_name}"
        engine = create_engine(db_url)
        today = date.today()
 
        with engine.begin() as conn:
            # Check existing row
            result = conn.execute(text("""
                SELECT request_count
                FROM user_request_limits
                WHERE user_id = :uid AND request_date = :today
            """), {"uid": user_id, "today": today}).fetchone()
 
            logger.info(f"User {user_id} on {today}: Found request_count={result[0] if result else None}")
 
            if result:
                count = result[0]
                if count >= limit:
                    logger.warning(f"User {user_id} reached daily limit of {limit} requests")
                    return False, 0  # blocked
                else:
                    conn.execute(text("""
                        UPDATE user_request_limits
                        SET request_count = request_count + 1
                        WHERE user_id = :uid AND request_date = :today
                    """), {"uid": user_id, "today": today})
                    logger.info(f"User {user_id} incremented request_count to {count + 1}")
                    return True, limit - (count + 1)
            else:
                # First request today
                conn.execute(text("""
                    INSERT INTO user_request_limits (user_id, request_date, request_count)
                    VALUES (:uid, :today, 1)
                """), {"uid": user_id, "today": today})
                logger.info(f"User {user_id} created new request limit entry with count=1")
                return True, limit - 1
    except Exception as e:
        logger.error(f"Error in check_and_increment_request_limit for user {user_id}: {str(e)}")
        return False, -1  # On error, block requests for safety

# Modify init_db to initialize SQLite tables and MySQL user_request_limits
def init_db(db_credentials: DatabaseCredentials):
    try:
        # Initialize SQLite tables (users, user_queries)
        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                prompt TEXT NOT NULL,
                sql_query TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            )
        """)
        conn.commit()
        logger.info("SQLite user database initialized successfully")
        conn.close()

        # Initialize user_request_limits in MySQL
        init_mysql_request_limits(db_credentials)
    except Exception as e:
        logger.error(f"Failed to initialize databases: {str(e)}")
        raise

def enhance_prompt_detailed(prompt: str, schema: str) -> dict:
    """
    Enhances a natural language query for SQL generation based on the provided schema.

    Args:
        prompt (str): The original user query.
        schema (str): The database schema as text.

    Returns:
        dict: A dictionary containing the enhanced prompt and metadata.
    """
    try:
        # Validate inputs
        if not prompt.strip():
            logger.error("Empty prompt provided")
            return {
                "enhanced_prompt": prompt,
                "improvements_made": [],
                "confidence_score": 0,
                "suggested_tables": [],
                "suggested_columns": [],
                "query_type": "SELECT",
                "explanation": "No enhancement possible: empty prompt provided",
                "success": False
            }
        if not schema.strip():
            logger.error("Empty schema provided, using default schema")
            schema = schema_to_text(DEFAULT_SCHEMA)

        # Extract schema information
        schema_info = extract_schema_info(schema)
        if not schema_info:
            logger.warning("No tables extracted from schema, using default schema")
            schema = schema_to_text(DEFAULT_SCHEMA)
            schema_info = extract_schema_info(schema)

        # Create schema summary with relationships
        schema_summary = []
        for table, columns in schema_info.items():
            schema_summary.append(f"Table {table}: {', '.join(columns)}")
            # Include foreign key information if available
            if table in DEFAULT_SCHEMA and DEFAULT_SCHEMA[table].get("foreign_keys"):
                for fk in DEFAULT_SCHEMA[table]["foreign_keys"]:
                    schema_summary.append(f"  Foreign Key: {fk['column']} -> {fk['referred_table']}({fk['referred_columns']})")
        schema_summary = "\n".join(schema_summary) if schema_summary else "No schema available"

        # Enhanced LLM prompt with few-shot examples
        enhancement_prompt = f"""
You are an expert at improving natural language queries for SQL generation, specializing in medical databases.

**Original User Query**: "{prompt}"

**Available Database Schema**:
{schema_summary}

**Instructions**:
Analyze the query and enhance it to be more specific, clear, and SQL-friendly while preserving the user's intent. Follow these rules:
1. **Clarity**: Rephrase vague terms (e.g., "show me data" → "retrieve records") and specify filters or aggregations.
2. **Schema Alignment**: Use table and column names from the schema (e.g., prefer 'cases', 'patients', 'case_answers').
3. **Medical Terms**: Handle synonyms and acronyms (e.g., 'diabetes' or 'DM', 'hypertension' or 'HTN').
4. **Query Type**: Identify the query type (SELECT, COUNT, AGGREGATE, COMPARISON, GROUP_BY).
5. **Minimal Changes**: Keep simple queries simple; don't overcomplicate.
6. **Table/Column Suggestions**: Suggest relevant tables and columns based on the schema and query context.

**Few-Shot Examples**:
1. **Input**: "Patients with diabetes"
   - **Enhanced Prompt**: "Retrieve cases where patients have a confirmed diabetes diagnosis."
   - **Improvements**: Specified 'cases' table, clarified 'diabetes diagnosis'.
   - **Suggested Tables**: ["cases", "case_answers", "questions"]
   - **Suggested Columns**: ["case_answers.answer_value", "questions.question_text"]
   - **Query Type**: SELECT
   - **Explanation**: Clarified intent to query cases with a specific condition, aligned with schema.

2. **Input**: "Average BMI"
   - **Enhanced Prompt**: "Calculate the average BMI for all patients from case answers."
   - **Improvements**: Added aggregation ('average'), specified 'case_answers' table.
   - **Suggested Tables**: ["cases", "case_answers", "questions"]
   - **Suggested Columns**: ["case_answers.answer_value", "questions.question_type"]
   - **Query Type**: AGGREGATE
   - **Explanation**: Specified aggregation and schema tables for precise SQL generation.

3. **Input**: "Show me high creatinine patients"
   - **Enhanced Prompt**: "Retrieve cases where patients have a serum creatinine value greater than 4."
   - **Improvements**: Specified numeric comparison, used medical term 'serum creatinine'.
   - **Suggested Tables**: ["cases", "case_answers", "questions"]
   - **Suggested Columns**: ["case_answers.answer_value", "questions.question_text"]
   - **Query Type**: COMPARISON
   - **Explanation**: Added numeric threshold and schema alignment for clarity.

**Medical Term Synonyms**:
- Diabetes: diabetes, diabetic, diabetes mellitus, DM
- Hypertension: hypertension, high blood pressure, HTN
- Creatinine: creatinine, serum creatinine, kidney function
- BMI: BMI, body mass index

**Output Format** (return valid JSON only):
{{
    "enhanced_prompt": "Improved query",
    "improvements_made": ["Improvement 1", "Improvement 2"],
    "confidence_score": 8,
    "suggested_tables": ["table1", "table2"],
    "suggested_columns": ["col1", "col2"],
    "query_type": "SELECT/COUNT/AGGREGATE/COMPARISON/GROUP_BY",
    "explanation": "Why these improvements help SQL generation"
}}
"""
        # Initialize LLM client with fallback model
        try:
            response = llm.chat.completions.create(
                model="openai/gpt-oss-20b",  # Primary model
                messages=[
                    {"role": "system", "content": "You are a SQL query enhancement expert. Respond with valid JSON only."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                max_tokens=500,
                temperature=0.9  # Slightly higher temperature for creative enhancements
            )
        except Exception as e:
            logger.warning(f"Primary model failed: {str(e)}. Falling back to gpt-4o-mini.")
            response = llm.chat.completions.create(
                model="openai/gpt-oss-20b",  # Fallback model
                messages=[
                    {"role": "system", "content": "You are a SQL query enhancement expert. Respond with valid JSON only."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                max_tokens=500,
                temperature=0.9
            )

        # Parse LLM response
        result_text = response.choices[0].message.content.strip()
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            return {
                "enhanced_prompt": prompt,
                "improvements_made": ["Attempted basic clarity improvements"],
                "confidence_score": 3,
                "suggested_tables": list(schema_info.keys()),
                "suggested_columns": [col for table in schema_info for col in schema_info[table]],
                "query_type": "SELECT",
                "explanation": f"Failed to parse LLM response: {str(e)}. Used fallback response.",
                "success": False
            }

        # Validate and normalize response
        required_fields = {
            "enhanced_prompt": prompt,
            "improvements_made": [],
            "confidence_score": 5,
            "suggested_tables": [],
            "suggested_columns": [],
            "query_type": "SELECT",
            "explanation": "No specific improvements identified"
        }
        for field, default in required_fields.items():
            if field not in result or result[field] is None:
                result[field] = default

        # Ensure query_type is valid
        valid_query_types = ["SELECT", "COUNT", "AGGREGATE", "COMPARISON", "GROUP_BY"]
        if result["query_type"] not in valid_query_types:
            result["query_type"] = "SELECT"
            result["improvements_made"].append("Corrected invalid query_type to SELECT")

        # Filter suggested tables and columns to those in schema
        result["suggested_tables"] = [t for t in result["suggested_tables"] if t in schema_info]
        result["suggested_columns"] = [c for c in result["suggested_columns"] if any(c in schema_info[t] for t in schema_info)]

        return {
            "enhanced_prompt": result["enhanced_prompt"],
            "improvements_made": result["improvements_made"],
            "confidence_score": min(max(int(result["confidence_score"]), 0), 10),  # Clamp between 0 and 10
            "suggested_tables": result["suggested_tables"],
            "suggested_columns": result["suggested_columns"],
            "query_type": result["query_type"],
            "explanation": result["explanation"],
            "success": True
        }
    except Exception as e:
        logger.error(f"Error in enhance_prompt_detailed: {str(e)}")
        return {
            "enhanced_prompt": prompt,
            "improvements_made": ["Basic grammar and clarity improvements"],
            "confidence_score": 3,
            "suggested_tables": list(schema_info.keys()) if 'schema_info' in locals() else [],
            "suggested_columns": [col for table in schema_info for col in schema_info[table]] if 'schema_info' in locals() else [],
            "query_type": "SELECT",
            "explanation": f"Fallback due to error: {str(e)}",
            "success": False
        }

def build_pdf_report(df: pd.DataFrame, title: str = "Report") -> bytes:
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        elements.append(Paragraph(title, styles['Title']))
        data = [df.columns.tolist()] + df.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)
        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

def generate_sql(schema: str, prompt: str) -> str:
    if not schema:
        logger.error("Empty schema provided to generate_sql, using default schema")
        schema = schema_to_text(DEFAULT_SCHEMA)
    
    schema_info = extract_schema_info(schema)
    if not schema_info:
        logger.error("No tables extracted from schema, using default schema")
        schema = schema_to_text(DEFAULT_SCHEMA)
        schema_info = extract_schema_info(schema)
    
    # schema_trimmed = filter_schema(schema, prompt)
    prompt_lower = prompt.lower()
    relevant_tables = []
    relevant_columns = {}

    medical_terms = {
            'diabetes': ['diabetes', 'diabetic', 'diabetes mellitus', 'dm', 'type 1 diabetes', 'type 2 diabetes'],
            'hypertension': ['hypertension', 'high blood pressure', 'htn', 'bp'],
            'heart': ['heart disease', 'cardiac', 'cardiovascular', 'coronary artery disease', 'heart failure'],
            'surgery': ['surgery', 'surgical', 'operative', 'operation', 'procedure'],
            'creatinine': ['creatinine', 'kidney function', 'renal function', 'serum creatinine'],
            'bmi': ['bmi', 'body mass index'],
            'gender': ['gender', 'sex', 'male', 'female'],
            'patient': ['patient', 'patients', 'case', 'cases', 'individual', 'person'],
            'blood group': ['blood group', 'blood type', 'abo', 'rhesus'],
            'trial': ['trial', 'clinical trial', 'study', 'research'],
            'age': ['age', 'years old'],
            'hospital': ['hospital', 'hospitals', 'general hospital', 'medical center', 'clinic']
        }

# Expand prompt keywords with synonyms
    prompt_lower = prompt.lower()
    prompt_keywords = set(prompt_lower.split())
    for word in prompt_lower.split():
        for term, synonyms in medical_terms.items():
            if word in synonyms:
                prompt_keywords.update(synonyms)

# Identify relevant tables and columns
    relevant_tables = []
    relevant_columns = {}
    for table, columns in schema_info.items():
        table_lower = table.lower()
        if any(keyword in table_lower for keyword in prompt_keywords) or any(
            any(keyword in col.lower() for keyword in prompt_keywords) for col in columns
        ):
            relevant_tables.append(table)
            relevant_columns[table] = [
                col for col in columns if any(keyword in col.lower() for keyword in prompt_keywords)
            ]

# Fallback: Include core tables that exist in schema
    if not relevant_tables:
        logger.warning(f"No relevant tables found for prompt: {prompt}. Including core tables as fallback.")
        core_tables = ['cases', 'patients', 'case_answers', 'questions', 'hospitals']
        relevant_tables = [table for table in schema_info.keys() if table.lower() in core_tables]
        relevant_columns = {table: schema_info[table] for table in relevant_tables}
        if not relevant_tables:
            logger.error("No core tables found in schema, using default schema")
            schema = schema_to_text(DEFAULT_SCHEMA)
            schema_info = extract_schema_info(schema)
            relevant_tables = list(schema_info.keys())
            relevant_columns = schema_info
        logger.info(f"Fallback tables included: {relevant_tables}")

    schema_summary = "\n".join([f"Table {table}: {', '.join(columns)}" for table, columns in schema_info.items()])

    sql_prompt = f"""
You are an expert SQL generator for MySQL 8.0. The current date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

**Database Structure Overview:**
- `cases`: Central table for patient encounters (id, patient_id, hospital_id, doctor_id, abstractor_id, status enum('draft','assigned','in_progress','completed','submitted','overdue'), priority enum('low','medium','high','urgent'), form_type enum('simple','advanced'), due_date, etc.).
- `questions`: Contains question details (id, category_id, question_text, question_type enum('text','number','date','radio','checkbox','dropdown','textarea'), is_required, display_order, parent_question_id, has_sub_questions, parent_options_question_id, has_sub_option_questions).
- `case_answers`: Stores answers (id, case_id, question_id, answer_value as string). Format of answer_value depends on question_type:
  - 'text'/'textarea': Free text (e.g., 'Trial A').
  - 'number': String representation of a number (e.g., '175').
  - 'date': String in 'YYYY-MM-DD' format (e.g., '2025-08-11').
  - 'radio'/'dropdown': Single option_value (e.g., 'female', 'a_positive').
  - 'checkbox': JSON array of option_values (e.g., '["serum_creatinine", "hemoglobin"]').
  - Rarely, JSON objects (e.g., '{{ "weight":"75","bmi":"2.4" }}') for structured sub-answers.
- `question_options`: For radio/checkbox/dropdown questions, maps option_text (human-readable, e.g., 'A positive') to option_value (slug, e.g., 'a_positive').
- `question_categories`: Groups questions (id, parent_category_id, name, e.g., 'Demographics', 'Risk factors'), with hierarchy via parent_category_id.
- `patients`, `doctors`, `hospitals`, `abstractors`, `hospital_managers`, `users`, `roles`, `doctor_abstractors`: Linked via respective IDs.

**Schema Provided:**
{schema}

**User Prompt:**
{prompt}

**CRITICAL RULES FOR GENERATING QUERIES:**

**1. SCHEMA-First TYPE DETECTION (MOST IMPORTANT):**
NEVER assume question_type from the prompt alone. ALWAYS follow this process:
- Step 1: Find potential matching questions in the schema by looking for question_text that contains relevant keywords.
- Step 2: Check the actual question_type for those questions in the schema.
- Step 3: Use the schema-confirmed question_type, NOT what you might infer from the prompt.

**COMMON MISTAKES TO AVOID:**
- Don't assume 'number' type just because the prompt mentions "value", "amount", "level", etc.
- Don't assume 'radio' type just because it sounds like yes/no.
- Don't assume JSON object format (e.g., '{{ "bmi":"25.5" }}') for numeric fields like BMI unless schema confirms; prefer 'number' type for metrics like BMI, age, or creatinine.
- ALWAYS verify against the schema first.

**Example of correct approach:**
If prompt asks "average BMI for patients with high creatinine levels" and schema shows:
- "BMI" (type: number, id=310) → Use CAST(ca.answer_value AS DECIMAL(10,2)) for aggregation.
- "Creatinine level high?" (type: radio, id=200) → Use ca.answer_value = 'yes'.
- "Highest Serum Creatinine value" (type: number, id=199) → Use only if prompt specifies numeric comparison (e.g., "> 4").


**2. AVOID DUPLICATE ROWS USING `EXISTS` FOR CONDITIONS:**
When checking if a case has a certain condition (e.g., "has diabetes"), DO NOT use a direct `JOIN` with `case_answers` in the main query unless retrieving answer values for aggregation (e.g., AVG). Use `EXISTS` subquery to prevent duplicates.

- **Correct Pattern for Conditions:**
SELECT c.*
FROM cases c
WHERE EXISTS (
    SELECT 1
    FROM case_answers ca
    JOIN questions q ON ca.question_id = q.id
    WHERE ca.case_id = c.id
        AND (LOWER(q.question_text) LIKE '%diabetes%' 
             OR LOWER(q.question_text) LIKE '%diabetic%' 
             OR LOWER(q.question_text) REGEXP '(^|[^a-z])dm([^a-z]|$)')
        AND q.question_type = 'radio'
        AND ca.answer_value = 'yes'
);

- **Correct Pattern for Aggregations (e.g., AVG):**
For aggregations like AVG, JOIN `case_answers` and `questions` in the main query for the aggregated field, and use `EXISTS` for additional conditions:
SELECT AVG(CAST(ca_agg.answer_value AS DECIMAL(10,2))) AS Average_BMI
FROM cases c
JOIN case_answers ca_agg ON c.id = ca_agg.case_id
JOIN questions q_agg ON ca_agg.question_id = q_agg.id
WHERE LOWER(q_agg.question_text) LIKE '%bmi%'
  AND q_agg.question_type = 'number'
  AND EXISTS (
    SELECT 1
    FROM case_answers ca
    JOIN questions q ON ca.question_id = q.id
    WHERE ca.case_id = c.id
        AND LOWER(q.question_text) LIKE '%diabetes%'
        AND q.question_type = 'radio'
        AND ca.answer_value = 'yes'
  );

**3. FLEXIBLE BUT PRECISE KEYWORD MATCHING:**
Use `LOWER(q.question_text)` for case-insensitive matching. Prioritize exact or near-exact matches to avoid confusion between similar questions (e.g., "Creatinine measured?" vs. "Highest Serum Creatinine value").
- Example: For "creatinine value > 4", prefer question_text like "Highest Serum Creatinine value" (number) over "Creatinine measured?" (radio).
- Use multiple keywords for better precision:
  - For "chronic lung disease": (LOWER(q.question_text) LIKE '%chronic%' AND LOWER(q.question_text) LIKE '%lung%')
- For acronyms (e.g., DM, HTN), use `REGEXP` to match whole words:
  - Correct: `LOWER(q.question_text) REGEXP '(^|[^a-z])dm([^a-z]|$)'`
  - Avoid: `'%dm%'` (causes false positives).

**4. TYPE-SPECIFIC QUERY PATTERNS:**
Based on the schema-confirmed question_type:

**A) Radio/Dropdown Questions (type = 'radio' or 'dropdown'):**
- Find the `questions.id` using flexible keyword matching inside an `EXISTS` subquery.
- Filter where `answer_value` is an exact affirmative (usually 'yes') or a specific option_value from `question_options`.
- If prompt specifies an option (e.g., "A positive"), JOIN `question_options` to map `option_text` to `option_value`.
- **Example:** For "patients with blood group A positive":
SELECT c.* 
FROM cases c 
WHERE EXISTS (
    SELECT 1 
    FROM case_answers ca 
    JOIN questions q ON ca.question_id = q.id 
    JOIN question_options o ON q.id = o.question_id
    WHERE ca.case_id = c.id 
        AND LOWER(q.question_text) LIKE '%blood group%'
        AND q.question_type IN ('radio', 'dropdown') 
        AND LOWER(o.option_text) LIKE '%a positive%'
        AND ca.answer_value = o.option_value
);

**B) Checkbox Questions (type = 'checkbox'):**
- The `answer_value` is a JSON-like string array (e.g., '["serum_creatinine", "hemoglobin"]').
- Use `JSON_CONTAINS` with `JSON_QUOTE` to check if a specific option was selected.
- **Example:** For cases where "Serum Creatinine" was selected in "Post operative labs":
SELECT c.* 
FROM cases c 
WHERE EXISTS (
    SELECT 1 
    FROM case_answers ca 
    JOIN questions q ON ca.question_id = q.id 
    JOIN question_options o ON q.id = o.question_id
    WHERE ca.case_id = c.id 
        AND LOWER(q.question_text) LIKE '%post operative labs%' 
        AND q.question_type = 'checkbox' 
        AND LOWER(o.option_text) LIKE '%serum creatinine%'
        AND JSON_CONTAINS(ca.answer_value, JSON_QUOTE(o.option_value))
);

**C) Number Questions (type = 'number'):**
- The `answer_value` is a string that must be converted.
- **ALWAYS `CAST` the `answer_value`** before comparing. Use `CAST(ca.answer_value AS DECIMAL(10,2))` for decimals (e.g., BMI, creatinine) or `AS SIGNED` for integers (e.g., age).
- Validate numeric data with `REGEXP '^[0-9]+\\.?[0-9]*$'` if needed to avoid invalid casts.
- **Example:** For patients with a creatinine value over 4:
SELECT c.* 
FROM cases c 
WHERE EXISTS (
    SELECT 1 
    FROM case_answers ca 
    JOIN questions q ON ca.question_id = q.id 
    WHERE ca.case_id = c.id 
        AND LOWER(q.question_text) LIKE '%serum creatinine%' 
        AND q.question_type = 'number' 
        AND CAST(ca.answer_value AS DECIMAL(10,2)) > 4
);

**D) Date Questions (type = 'date'):**
- Use `STR_TO_DATE(ca.answer_value, '%Y-%m-%d')` for comparisons.
- **Example:** For surgeries after 2025-01-01:
SELECT c.* 
FROM cases c 
WHERE EXISTS (
    SELECT 1 
    FROM case_answers ca 
    JOIN questions q ON ca.question_id = q.id 
    WHERE ca.case_id = c.id 
        AND LOWER(q.question_text) LIKE '%surgery%date%' 
        AND q.question_type = 'date' 
        AND STR_TO_DATE(ca.answer_value, '%Y-%m-%d') > '2025-01-01'
);

**E) Text/Textarea Questions (type = 'text' or 'textarea'):**
- Use `ca.answer_value LIKE '%keyword%'` for partial matches.
- **Example:** For cases with trial name containing "Trial A":
SELECT c.* 
FROM cases c 
WHERE EXISTS (
    SELECT 1 
    FROM case_answers ca 
    JOIN questions q ON ca.question_id = q.id 
    WHERE ca.case_id = c.id 
        AND LOWER(q.question_text) LIKE '%trial%' 
        AND q.question_type IN ('text', 'textarea') 
        AND ca.answer_value LIKE '%Trial A%'
);

**F) JSON Object Answers:**
- For rare JSON objects (e.g., '{{ "weight":"75" }}'), use `JSON_EXTRACT`.
- Verify schema confirms JSON object format; avoid assuming JSON for numeric fields like BMI unless specified (id=264 in schema).
- **Example:** For cases with weight = 75:
SELECT c.* 
FROM cases c 
WHERE EXISTS (
    SELECT 1 
    FROM case_answers ca 
    JOIN questions q ON ca.question_id = q.id 
    WHERE ca.case_id = c.id 
        AND LOWER(q.question_text) LIKE '%weight%' 
        AND JSON_EXTRACT(ca.answer_value, '$.weight') = '75'
);

**5. AGGREGATIONS (e.g., AVG, COUNT):**
- For aggregations like AVG, JOIN `case_answers` and `questions` in the main query for the aggregated field, using distinct aliases (e.g., `ca_agg`, `q_agg`).
- Use `EXISTS` for additional conditions to avoid duplicates.
- For numeric aggregations, prefer `number` type questions and use `CAST(ca_agg.answer_value AS DECIMAL(10,2))`.
- **Example:** For average BMI of patients with diabetes:
SELECT AVG(CAST(ca_agg.answer_value AS DECIMAL(10,2))) AS Average_BMI
FROM cases c
JOIN case_answers ca_agg ON c.id = ca_agg.case_id
JOIN questions q_agg ON ca_agg.question_id = q_agg.id
WHERE LOWER(q_agg.question_text) LIKE '%bmi%'
  AND q_agg.question_type = 'number'
  AND EXISTS (
    SELECT 1
    FROM case_answers ca
    JOIN questions q ON ca.question_id = q.id
    WHERE ca.case_id = c.id
        AND LOWER(q.question_text) LIKE '%diabetes%'
        AND q.question_type = 'radio'
        AND ca.answer_value = 'yes'
  );

**6. WHEN SCHEMA INFORMATION IS UNCLEAR:**
If you cannot determine the exact question_type from the schema:
- Use a flexible approach that handles multiple possible types.
- Prioritize the most common patterns: 'radio' for yes/no, 'number' for numeric values, 'checkbox' for multiple selections.
- Add comments in the query explaining the assumption.
- **Example:**
SELECT c.*
FROM cases c
WHERE EXISTS (
    SELECT 1 
    FROM case_answers ca 
    JOIN questions q ON ca.question_id = q.id 
    WHERE ca.case_id = c.id 
        AND LOWER(q.question_text) LIKE '%creatinine%'
        AND (
            (q.question_type = 'radio' AND ca.answer_value = 'yes') 
            OR (q.question_type = 'number' AND CAST(ca.answer_value AS DECIMAL(10,2)) > 0)
        )
) -- Assuming radio or number type based on context;

**7. JOIN OTHER TABLES WHEN NEEDED:**
- Join `patients`, `doctors`, `hospitals`, etc., for additional data (e.g., patient name, hospital location).
- Join `question_categories` for category-based queries (e.g., questions in "Demographics").
- For sub-questions, filter on `questions.parent_question_id` or use a self-join.



**8. COMMON MEDICAL TERM VARIATIONS AND ACRONYMS:**
- Diabetes: diabetes, diabetic, diabetes mellitus, `REGEXP '(^|[^a-z])dm([^a-z]|$)'`
- Hypertension: hypertension, high blood pressure, `REGEXP '(^|[^a-z])htn([^a-z]|$)'`
- Heart conditions: heart disease, cardiac, cardiovascular, coronary artery disease
- Surgery: surgery, surgical, operative, operation, procedure
- Creatinine: creatinine, kidney function, renal function
- BMI: bmi, body mass index
- Avoid false positives with regex for acronyms.


**9. HANDLING AMBIGUOUS PROMPTS:**
- If the prompt suggests a numeric comparison (e.g., "creatinine value > 4") but the question is radio (e.g., "Creatinine measured?"), prioritize `q.question_type = 'radio'` and match `answer_value = 'yes'` for presence, not numeric comparison.
- If multiple questions match (e.g., "creatinine" in radio and number questions), prefer the most specific `question_text` match and validate type via schema.
- For aggregations (e.g., "average BMI"), assume `number` type unless schema confirms JSON object format.

**10. DATA INSIGHTS (SCHEMA-SPECIFIC EXAMPLES):**
- Radio: "Patient participating in any trial" (id=204, yes/no), "Blood Group" (id=300, a_positive).
- Checkbox: "Post operative labs" (id=198, serum_creatinine), "Family history of Coronary Artery Disease" (id=201, premature).
- Number: "Highest Serum Creatinine value" (id=199), "Age" (id=310), "BMI" (id=311).
- Text: "If participating in any trial yes, name of the trial" (id=205).
- JSON objects: '{{ "weight":"75" }}' (id=264).

**11. FEW-SHOT EXAMPLES:**
- **Request: "Patients with diabetes"**
SELECT c.*
FROM cases c
WHERE EXISTS (
    SELECT 1 
    FROM case_answers ca 
    JOIN questions q ON ca.question_id = q.id 
    WHERE ca.case_id = c.id 
        AND (LOWER(q.question_text) LIKE '%diabetes%'
             OR LOWER(q.question_text) LIKE '%diabetic%'
             OR LOWER(q.question_text) REGEXP '(^|[^a-z])dm([^a-z]|$)')
        AND q.question_type = 'radio'
        AND ca.answer_value = 'yes'
);

- **Request: "Cases with serum creatinine selected in post operative labs"**
SELECT c.*
FROM cases c
WHERE EXISTS (
    SELECT 1 
    FROM case_answers ca 
    JOIN questions q ON ca.question_id = q.id 
    JOIN question_options o ON q.id = o.question_id
    WHERE ca.case_id = c.id 
        AND LOWER(q.question_text) LIKE '%post operative labs%'
        AND q.question_type = 'checkbox'
        AND LOWER(o.option_text) LIKE '%serum creatinine%'
        AND JSON_CONTAINS(ca.answer_value, JSON_QUOTE(o.option_value))
);

- **Request: "Cases with creatinine value greater than 4"**
SELECT c.*
FROM cases c
WHERE EXISTS (
    SELECT 1 
    FROM case_answers ca 
    JOIN questions q ON ca.question_id = q.id 
    WHERE ca.case_id = c.id 
        AND LOWER(q.question_text) LIKE '%serum creatinine%'
        AND q.question_type = 'number'
        AND CAST(ca.answer_value AS DECIMAL(10,2)) > 4
);

- **Request: "Cases where creatinine was measured"**
SELECT c.*
FROM cases c
WHERE EXISTS (
    SELECT 1 
    FROM case_answers ca 
    JOIN questions q ON ca.question_id = q.id 
    WHERE ca.case_id = c.id 
        AND LOWER(q.question_text) LIKE '%creatinine measured%'
        AND q.question_type = 'radio'
        AND ca.answer_value = 'yes'
);

- **Request: "Cases with age > 50"**
SELECT c.*
FROM cases c
WHERE EXISTS (
    SELECT 1 
    FROM case_answers ca 
    JOIN questions q ON ca.question_id = q.id 
    WHERE ca.case_id = c.id 
        AND LOWER(q.question_text) LIKE '%age%'
        AND q.question_type = 'number'
        AND CAST(ca.answer_value AS SIGNED) > 50
);

- **Request: "Cases with trial name containing 'Trial A'"**
SELECT c.*
FROM cases c
WHERE EXISTS (
    SELECT 1 
    FROM case_answers ca 
    JOIN questions q ON ca.question_id = q.id 
    WHERE ca.case_id = c.id 
        AND LOWER(q.question_text) LIKE '%trial%'
        AND q.question_type IN ('text', 'textarea')
        AND ca.answer_value LIKE '%Trial A%'
);

- **Request: "Average BMI of patients with family history of premature Coronary Artery Disease"**
SELECT AVG(CAST(ca_agg.answer_value AS DECIMAL(10,2))) AS Average_BMI
FROM cases c
JOIN case_answers ca_agg ON c.id = ca_agg.case_id
JOIN questions q_agg ON ca_agg.question_id = q_agg.id
WHERE LOWER(q_agg.question_text) LIKE '%bmi%'
  AND q_agg.question_type = 'number'
  AND EXISTS (
    SELECT 1
    FROM case_answers ca
    JOIN questions q ON ca.question_id = q.id
    JOIN question_options o ON q.id = o.question_id
    WHERE ca.case_id = c.id
      AND LOWER(q.question_text) LIKE '%family history%'
      AND LOWER(q.question_text) LIKE '%coronary artery disease%'
      AND q.question_type = 'checkbox'
      AND LOWER(o.option_text) LIKE '%premature%'
      AND JSON_CONTAINS(ca.answer_value, JSON_QUOTE(o.option_value))
  );

**12. GENERAL MySQL 8.0 SYNTAX:**
- Always generate syntactically valid MySQL 8.0 SQL.
- Just return the query alone. Do not include markdown fences like ```sql.
- Use JSON functions (JSON_CONTAINS, JSON_EXTRACT, JSON_QUOTE) for JSON handling.
- Ensure proper joins, table aliases (e.g., `ca_agg`, `q_agg` for aggregations), and NULL handling.
- For aggregations, validate that `answer_value` is accessed from the correct table alias to avoid errors like "Unknown column 'ca.answer_value'".

**13. Dynamic Grouping for String Fields**:
   - If the prompt implies grouping (e.g., 'count by', 'distribution of', 'visualize'), identify the target question and group by:
     - `case_answers.answer_value` for 'text', 'radio', or 'dropdown'.
     - `question_options.option_text` for 'radio' or 'dropdown' to get human-readable labels.
   - Example: For "count patients by gender":
     SELECT o.option_text AS Label, COUNT(*) AS Count
     FROM cases c
     JOIN case_answers ca ON c.id = ca.case_id
     JOIN questions q ON ca.question_id = q.id
     JOIN question_options o ON q.id = o.question_id AND ca.answer_value = o.option_value
     WHERE LOWER(q.question_text) LIKE '%gender%'
       AND q.question_type IN ('radio', 'dropdown')
     GROUP BY o.option_text;

**14. Chart-Friendly Output**:
   - For prompts requesting visualization (e.g., 'visualize', 'chart', 'graph'), return two columns:
     - First column: The string value (e.g., `option_text` or `answer_value`).
     - Second column: The count (e.g., COUNT(*)).
   - Ensure the query is compatible with bar or pie charts.

**15. Handle Ambiguous Prompts**:
   - If the prompt is unclear, assume a grouping query for string fields if keywords like 'by', 'distribution', or 'visualize' are present.
   - Prioritize 'radio' or 'dropdown' for categorical data unless schema suggests 'text' or 'checkbox'.

"""
    try:
        response = llm.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a SQL generator. Respond with a valid MySQL query enclosed in triple backticks (```). Do not include explanations or additional text."},
                {"role": "user", "content": sql_prompt}
            ],
            temperature=0.3
        )
        sql_query = response.choices[0].message.content.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

        # Validate and clean the SQL to remove duplicate GROUP BY clauses
        sql_lines = sql_query.split('\n')
        cleaned_lines = []
        group_by_seen = False
        for line in sql_lines:
            line = line.strip()
            if line.lower().startswith("group by") and not group_by_seen:
                cleaned_lines.append(line)
                group_by_seen = True
            elif not line.lower().startswith("group by") or not group_by_seen:
                cleaned_lines.append(line)
        sql_query = "\n".join(cleaned_lines)

        if not sql_query.strip():
            logger.error("Generated SQL query is empty")
            raise HTTPException(status_code=400, detail="Generated SQL query is empty")

        logger.info(f"Generated SQL: {sql_query}")
        logger.info(f"Relevant tables identified: {relevant_tables}")
        logger.info(f"Relevant columns: {relevant_columns}")

        return sql_query
    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        raise HTTPException(status_code=400, detail=f"SQL generation failed: {str(e)}")

@app.post("/api/bootstrap-schema")
async def bootstrap_schema_endpoint(request: BootstrapSchemaRequest):
    try:
        # Initialize user_request_limits table in MySQL
        init_mysql_request_limits(request.db_credentials)
        schema = bootstrap_schema(request.db_credentials)
        return {"schema": schema}
    except Exception as e:
        logger.error(f"Error in bootstrap-schema endpoint: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# @app.post("/api/enhance-prompt")
# async def enhance_prompt(request: EnhancePromptRequest, user_id: Optional[int] = Depends(get_user_id_from_token)):
#     try:
#         # Validate query
#         if not request.query.strip():
#             logger.error("Empty query provided to enhance-prompt endpoint")
#             raise HTTPException(status_code=400, detail="Query cannot be empty")

#         # Load schema
#         schema = request.schema_text or schema_to_text(load_schema(request.db_credentials))
#         if not schema.strip():
#             logger.error("Empty schema loaded for enhance-prompt endpoint")
#             raise HTTPException(status_code=400, detail="Schema is empty. Please bootstrap the schema.")

#         # Enhance prompt
#         result = enhance_prompt_detailed(request.query, schema)
        
#         # Save query to history
#         save_query(user_id, request.query, None)
        
#         return result
#     except HTTPException as e:
#         logger.error(f"HTTP Exception in enhance-prompt endpoint: {str(e)}")
#         raise e
#     except Exception as e:
#         logger.error(f"Error in enhance-prompt endpoint: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Prompt enhancement failed: {str(e)}")

def enhance_prompt_simple(prompt: str, schema: str) -> str:
    """
    Enhances a natural language query for SQL generation based on the provided schema, returning only the enhanced prompt.

    Args:
        prompt (str): The original user query.
        schema (str): The database schema as text.

    Returns:
        str: The enhanced prompt.
    """
    try:
        # Validate inputs
        if not prompt.strip():
            logger.error("Empty prompt provided")
            return prompt
        if not schema.strip():
            logger.error("Empty schema provided, using default schema")
            schema = schema_to_text(DEFAULT_SCHEMA)

        # Extract schema information
        schema_info = extract_schema_info(schema)
        if not schema_info:
            logger.warning("No tables extracted from schema, using default schema")
            schema = schema_to_text(DEFAULT_SCHEMA)
            schema_info = extract_schema_info(schema)

        # Create schema summary with relationships
        schema_summary = []
        for table, columns in schema_info.items():
            schema_summary.append(f"Table {table}: {', '.join(columns)}")
            if table in DEFAULT_SCHEMA and DEFAULT_SCHEMA[table].get("foreign_keys"):
                for fk in DEFAULT_SCHEMA[table]["foreign_keys"]:
                    schema_summary.append(f"  Foreign Key: {fk['column']} -> {fk['referred_table']}({fk['referred_columns']})")
        schema_summary = "\n".join(schema_summary) if schema_summary else "No schema available"

        # Enhanced LLM prompt with few-shot examples
        enhancement_prompt = f"""
You are an expert at improving natural language queries for SQL generation, specializing in medical databases.

**Original User Query**: "{prompt}"

**Available Database Schema**:
{schema_summary}

**Instructions**:
Analyze the query and enhance it to be more specific, clear, and SQL-friendly while preserving the user's intent. Return only the enhanced prompt as a string. Follow these rules:
1. **Clarity**: Rephrase vague terms (e.g., "show me data" → "retrieve records") and specify filters or aggregations.
2. **Schema Alignment**: Use table and column names from the schema (e.g., prefer 'cases', 'patients', 'case_answers').
3. **Medical Terms**: Handle synonyms and acronyms (e.g., 'diabetes' or 'DM', 'hypertension' or 'HTN').
4. **Minimal Changes**: Keep simple queries simple; don't overcomplicate.

1. Make the query more specific and SQL-friendly
2. Suggest relevant tables and columns from the schema
3. Add clarity about aggregations, filters, or grouping if needed
4. Maintain the user's original intent
5. Don't overly complicate simple requests
6. If the original is already good, make minimal changes
 
Focus on:
- Clarity and specificity
- Proper table/column references
- Clear aggregation requests (SUM, COUNT, AVG, etc.)
- Specific time periods or filters
- Proper grouping or sorting requests

**Medical Term Synonyms**:
- Diabetes: diabetes, diabetic, diabetes mellitus, DM
- Hypertension: hypertension, high blood pressure, HTN
- Creatinine: creatinine, serum creatinine, kidney function
- BMI: BMI, body mass index

**Output**: Return only the enhanced prompt as a string.
"""
        # Initialize LLM client with fallback model
        try:
            response = llm.chat.completions.create(
                model="deepseek/deepseek-chat-v3.1:free",  # Primary model
                messages=[
                    {"role": "system", "content": "You are a SQL query enhancement expert. Respond with the enhanced prompt as a string only."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                max_tokens=200,
                temperature=0.9
            )
        except Exception as e:
            logger.warning(f"Primary model failed: {str(e)}. Falling back to gpt-4o-mini.")
            response = llm.chat.completions.create(
                model="deepseek/deepseek-chat-v3.1:free",  # Fallback model
                messages=[
                    {"role": "system", "content": "You are a SQL query enhancement expert. Respond with the enhanced prompt as a string only."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                max_tokens=200,
                temperature=0.9
            )

        # Parse LLM response
        enhanced_prompt = response.choices[0].message.content.strip()
        return enhanced_prompt if enhanced_prompt else prompt

    except Exception as e:
        logger.error(f"Error in enhance_prompt_simple: {str(e)}")
        return prompt
       
@app.post("/api/enhance-prompt")
async def enhance_prompt(request: EnhancePromptRequest, user_id: Optional[int] = Depends(get_user_id_from_token)):
    try:
        # Validate query
        if not request.query.strip():
            logger.error("Empty query provided to enhance-prompt endpoint")
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Load schema
        schema = request.schema_text or schema_to_text(load_schema(request.db_credentials))
        if not schema.strip():
            logger.error("Empty schema loaded for enhance-prompt endpoint")
            raise HTTPException(status_code=400, detail="Schema is empty. Please bootstrap the schema.")

        # Enhance prompt
        enhanced_prompt = enhance_prompt_simple(request.query, schema)
        
        # Save query to history
        save_query(user_id, request.query, None)
        
        return {"enhanced_prompt": enhanced_prompt}
    except HTTPException as e:
        logger.error(f"HTTP Exception in enhance-prompt endpoint: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Error in enhance-prompt endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prompt enhancement failed: {str(e)}")

@app.post("/api/generate-sql")
async def generate_sql_endpoint(request: QueryRequest, user_id: Optional[int] = Depends(get_user_id_from_token)):
    try:
        allowed, remaining = check_and_increment_request_limit(user_id, request.db_credentials, limit=50)
        if not allowed:
                logger.warning(f"User {user_id} blocked: daily request limit (3) reached, {remaining} remaining")
                raise HTTPException(
                    status_code=429,
                    detail=f"Daily request limit (3) reached. {remaining} requests remaining. Please try again tomorrow."
                )
        schema_file_1 = get_schema_file_path(request.db_credentials)
        if not os.path.exists(schema_file_1):
            logger.error(f"Schema file {schema_file_1} does not exist.")
            raise HTTPException(status_code=400, detail="Schema file not found. Please bootstrap the schema.")

        schema = request.schema_text or schema_to_text(load_schema(request.db_credentials))
        if not schema.strip():
            logger.error("Schema text is empty.")
            raise HTTPException(status_code=400, detail="Schema is empty. Please bootstrap the schema.")

        effective_prompt = request.enhanced_prompt if request.use_enhanced else request.query
        schema_trimmed = filter_schema(schema, effective_prompt)
        if not schema_trimmed.strip():
            logger.warning("Trimmed schema is empty, falling back to full schema.")
            schema_trimmed = schema  # Fallback to full schema if trimmed schema is empty
        logger.info(f"Trimmed schema: {schema_trimmed}")
        sql_query = generate_sql(schema, effective_prompt)
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        sql_query = "\n".join(line.strip() for line in sql_query.splitlines() if line.strip())
        if not sql_query:
            logger.error("Generated SQL query is empty.")
            raise HTTPException(status_code=400, detail="Generated SQL query is empty")

        logger.info(f"Cleaned SQL query: {sql_query}")

        db_url = f"mysql+pymysql://{request.db_credentials.db_user}:{urllib.parse.quote_plus(request.db_credentials.db_password)}@{request.db_credentials.db_host}:{request.db_credentials.db_port}/{request.db_credentials.db_name}"
        try:
            engine = create_engine(db_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as conn_err:
            logger.error(f"Database connection test failed: {str(conn_err)}")
            raise HTTPException(status_code=500, detail=f"Database connection failed: {str(conn_err)}.")

        with engine.connect() as conn:
            conn.execute(text("SET SESSION sql_mode='NO_ENGINE_SUBSTITUTION,STRICT_TRANS_TABLES'"))
            result = conn.execute(text(sql_query))
            columns = list(result.keys())
            rows = [
                [float(val) if isinstance(val, decimal.Decimal) else val for val in row]
                for row in result.fetchall()
            ]

        response = {
            "sql": sql_query,
            "columns": columns,
            "rows": rows,
            "chart_type": request.chart_type,
            "chart_title": f"Report for {effective_prompt[:50]}",
            "chart_data": None,
            "box_plot_data": None,
            "chart_error": None,
            "export_data": None,
            "export_mime": None,
            "export_filename": None,
            "remaining_requests": remaining  # Add remaining requests to response
        }

        valid_chart_types = ['bar', 'pie', 'line', 'scatter', 'box', 'histogram']
        chart_keywords = ['chart', 'graph', 'visualize', 'plot', 'distribution']
        chart_requested = any(keyword in effective_prompt.lower() for keyword in chart_keywords) and request.chart_type in valid_chart_types

        # if chart_requested and len(columns) >= 1:
        #     df = pd.DataFrame(rows, columns=columns)
        #     for col in df.columns:
        #         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float, errors='ignore')

        #     label_column = columns[0]
        #     data_columns = [col for col in columns[1:] if pd.api.types.is_numeric_dtype(df[col])]

            # if request.chart_type == "box":
            #     box_plot_data = []
            #     try:
            #         if data_columns:
            #             data_column = data_columns[0]  # Use the first numeric column
            #             if len(columns) > 1 and pd.api.types.is_string_dtype(df[label_column]):
            #                 # Group by categorical column (e.g., department)
            #                 grouped = df.groupby(label_column)
            #                 for group_name, group_data in grouped:
            #                     numeric_series = group_data[data_column].dropna()
            #                     if not numeric_series.empty:
            #                         min_val = float(numeric_series.min())
            #                         q1 = float(numeric_series.quantile(0.25))
            #                         median = float(numeric_series.median())
            #                         q3 = float(numeric_series.quantile(0.75))
            #                         max_val = float(numeric_series.max())
            #                         iqr = q3 - q1
            #                         lower_bound = q1 - 1.5 * iqr
            #                         upper_bound = q3 + 1.5 * iqr
            #                         outliers = numeric_series[(numeric_series < lower_bound) | (numeric_series > upper_bound)].tolist()
            #                         box_plot_data.append({
            #                             "group": str(group_name),
            #                             "min": min_val,
            #                             "q1": q1,
            #                             "median": median,
            #                             "q3": q3,
            #                             "max": max_val,
            #                             "outliers": [float(x) for x in outliers]
            #                         })
            #             else:
            #                 # Single box plot for non-grouped data
            #                 numeric_series = df[data_column].dropna()
            #                 if not numeric_series.empty:
            #                     min_val = float(numeric_series.min())
            #                     q1 = float(numeric_series.quantile(0.25))
            #                     median = float(numeric_series.median())
            #                     q3 = float(numeric_series.quantile(0.75))
            #                     max_val = float(numeric_series.max())
            #                     iqr = q3 - q1
            #                     lower_bound = q1 - 1.5 * iqr
            #                     upper_bound = q3 + 1.5 * iqr
            #                     outliers = numeric_series[(numeric_series < lower_bound) | (numeric_series > upper_bound)].tolist()
            #                     box_plot_data.append({
            #                         "group": data_column,
            #                         "min": min_val,
            #                         "q1": q1,
            #                         "median": median,
            #                         "q3": q3,
            #                         "max": max_val,
            #                         "outliers": [float(x) for x in outliers]
            #                     })
            #             if box_plot_data:
            #                 response["box_plot_data"] = box_plot_data
            #                 response["chart_title"] = f"Box Plot of {data_column} by {label_column if len(columns) > 1 and pd.api.types.is_string_dtype(df[label_column]) else 'Data'}"
            #             else:
            #                 response["chart_error"] = "No valid numeric data for box plot."
            #         else:
            #             response["chart_error"] = "No numeric columns found for box plot."
            #     except Exception as e:
            #         logger.error(f"Box plot calculation error: {str(e)}")
            #         response["box_plot_data"] = []
            #         response["chart_error"] = f"Box plot generation failed: {str(e)}"


        if chart_requested and len(columns) >= 1:
                df = pd.DataFrame(rows, columns=columns)

    # Identify label and data columns
                label_column = columns[0]  # e.g., "hospital_group"
                data_columns = [col for col in columns[1:] if pd.to_numeric(df[col], errors='coerce').notna().any()]  # e.g., "height_cm"

    # Convert only data columns to numeric, preserving label column as string
                for col in data_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float, errors='ignore')

                label_column = columns[0]
                data_columns = [col for col in columns[1:] if pd.api.types.is_numeric_dtype(df[col])]

                if request.chart_type == "box":
                    box_plot_data = []
                    try:
                        if data_columns:
                            data_column = data_columns[0]  # Use the first numeric column
                            if len(columns) > 1 and pd.api.types.is_string_dtype(df[label_column]):
                    # Group by categorical column (e.g., department)
                                grouped = df.groupby(label_column)
                                for group_name, group_data in grouped:
                                    numeric_series = group_data[data_column].dropna()
                                    if not numeric_series.empty:
                                        min_val = float(numeric_series.min())
                                        q1 = float(numeric_series.quantile(0.25))
                                        median = float(numeric_series.median())
                                        q3 = float(numeric_series.quantile(0.75))
                                        max_val = float(numeric_series.max())
                                        iqr = q3 - q1
                                        lower_bound = q1 - 1.5 * iqr
                                        upper_bound = q3 + 1.5 * iqr
                                        outliers = numeric_series[(numeric_series < lower_bound) | (numeric_series > upper_bound)].tolist()
                                        box_plot_data.append({
                                "group": str(group_name),
                                "min": min_val,
                                "q1": q1,
                                "median": median,
                                "q3": q3,
                                "max": max_val,
                                "outliers": [float(x) for x in outliers]
                                })
                            else:
                    # Single box plot for non-grouped data
                                numeric_series = df[data_column].dropna()
                                if not numeric_series.empty:
                                    min_val = float(numeric_series.min())
                                    q1 = float(numeric_series.quantile(0.25))
                                    median = float(numeric_series.median())
                                    q3 = float(numeric_series.quantile(0.75))
                                    max_val = float(numeric_series.max())
                                    iqr = q3 - q1
                                    lower_bound = q1 - 1.5 * iqr
                                    upper_bound = q3 + 1.5 * iqr
                                    outliers = numeric_series[(numeric_series < lower_bound) | (numeric_series > upper_bound)].tolist()
                                    box_plot_data.append({
                                        "group": data_column,
                                        "min": min_val,
                                        "q1": q1,
                                        "median": median,
                                        "q3": q3,
                                        "max": max_val,
                                        "outliers": [float(x) for x in outliers]
                                        })
                            if box_plot_data:
                                response["box_plot_data"] = box_plot_data
                                response["chart_title"] = f"Box Plot of {data_column} by {label_column if len(columns) > 1 and pd.api.types.is_string_dtype(df[label_column]) else 'Data'}"
                            else:
                                response["chart_error"] = "No valid numeric data for box plot."
                        else:
                            response["chart_error"] = "No numeric columns found for box plot."
                    except Exception as e:
                        logger.error(f"Box plot calculation error: {str(e)}")
                        response["box_plot_data"] = []
                        response["chart_error"] = f"Box plot generation failed: {str(e)}"            


            # Handle export
        if request.export_format != "None":
                df = pd.DataFrame(rows, columns=columns)
                if request.export_format == "CSV":
                    if chart_requested and request.chart_type in valid_chart_types and response.get("chart_data"):
                        chart_df = pd.DataFrame({
                            "labels": response["chart_data"]["data"]["labels"],
                            "data": response["chart_data"]["data"]["datasets"][0]["data"]
                        })
                        csv_data = chart_df.to_csv(index=False)
                        response["export_data"] = csv_data
                        response["export_mime"] = "text/csv"
                        response["export_filename"] = "chart.csv"
                    elif request.chart_type == "box" and response.get("box_plot_data"):
                        box_df = pd.DataFrame(response["box_plot_data"])
                        csv_data = box_df.to_csv(index=False)
                        response["export_data"] = csv_data
                        response["export_mime"] = "text/csv"
                        response["export_filename"] = "box_plot.csv"
                    else:
                        csv_data = df.to_csv(index=False)
                        response["export_data"] = csv_data
                        response["export_mime"] = "text/csv"
                        response["export_filename"] = "export.csv"
                elif request.export_format == "PDF":
                    pdf_bytes = build_pdf_report(df, response["chart_title"])
                    response["export_data"] = pdf_bytes.hex()
                    response["export_mime"] = "application/pdf"
                    response["export_filename"] = "export.pdf"

        save_query(user_id, effective_prompt, sql_query)
        logger.info(f"SQL query executed successfully: {sql_query}")
        return response
    except sqlalchemy.exc.OperationalError as e:
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}.")
    except HTTPException as e:
        logger.error(f"HTTP Exception: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Error in generate-sql endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SQL execution failed: {str(e)}")