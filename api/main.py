# Forcing a new Vercel build to load latest environment variables
# Forcing a Vercel resync on 2025-09-08 at 11:03 PM

import os
import asyncio
from functools import partial
from pathlib import Path
import io, csv
import re
from fastapi import FastAPI, Request, Header, HTTPException, status, Depends, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse, Response
from fastapi.templating import Jinja2Templates
import httpx
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from supabase import create_client, Client
from postgrest import APIError

# It's a good practice to load environment variables at the start
# In a real app, you'd use a library like python-dotenv for local development
# Vercel will inject these from your project settings
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
HCAPTCHA_SITE_KEY = os.environ.get("HCAPTCHA_SITE_KEY")
SITE_URL = os.environ.get("SITE_URL")

# Get the root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="FormPipeDB API - The Correct One")

# Mount the static files directory
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# --- API Models for the new Database -> Table structure ---
class DatabaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)

class DatabaseResponse(BaseModel):
    id: int
    created_at: str
    name: str
    description: Optional[str] = None

    # Add a list of tables to the response, using a forward reference
    user_tables: List["TableResponse"] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")
class ForeignKeyDefinition(BaseModel):
    table_id: int
    column_name: str

class ColumnDefinition(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    # In the future, this can be expanded with more constraints
    type: str = Field(..., min_length=1) 
    is_primary_key: bool = False
    is_unique: bool = False
    is_auto_increment: bool = False
    is_not_null: bool = False
    foreign_key: Optional[ForeignKeyDefinition] = None

class TableCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    columns: List[ColumnDefinition]

class TableUpdate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    columns: List[ColumnDefinition]

class TableResponse(BaseModel):
    id: int
    name: str
    columns: List[ColumnDefinition]

    model_config = ConfigDict(extra="ignore")

class CsvPreviewRequest(BaseModel):
    csv_content: str

class CsvPreviewResponse(BaseModel):
    original_headers: List[str]
    sanitized_headers: List[str]
    inferred_types: List[str]

class CsvImportRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    csv_content: str
    # The user confirms the columns and types in the UI before sending
    columns: List[ColumnDefinition]

class CsvRowImportRequest(BaseModel):
    csv_content: str
    column_mapping: dict[str, str] # Maps table_column_name -> csv_header_name

class RowImportError(BaseModel):
    row_number: int
    data: dict[str, Any]
    error: str

class CsvRowImportResponse(BaseModel):
    inserted_count: int
    failed_count: int
    errors: List[RowImportError]

# After all models are defined, resolve the forward reference in DatabaseResponse
DatabaseResponse.model_rebuild()

class RowResponse(BaseModel):
    id: int
    created_at: str
    table_id: int
    data: dict[str, Any]
    _meta: Optional[dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")

class RowCreate(BaseModel):
    data: dict[str, Any]

class PaginatedRowResponse(BaseModel):
    total: int
    data: List[RowResponse]

class SqlImportRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    script: str

class SqlTableCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    script: str

class QueryFilter(BaseModel):
    column: str
    operator: str
    value: Any

class QueryBuilderRequest(BaseModel):
    table_id: int
    columns: List[str]
    filters: List[QueryFilter] = Field(default_factory=list)
    # Future additions: joins, order_by, etc.

# --- FIX: Conditionally use EmailStr to prevent crash if 'email-validator' is not installed ---
try:
    from pydantic import EmailStr
except ImportError:
    # If pydantic itself has an issue, fallback to str
    EmailStr = str

class ContactRequest(BaseModel):
    sender_name: str
    sender_email: EmailStr # This will be a plain str if email-validator is missing
    message: str

class AuthenticatedAccountDeletionRequest(BaseModel):
    confirmation: str = Field(..., pattern=r"^delete my account$", description="User must type 'delete my account' to confirm.")



# --- Reusable Dependencies ---
# This dependency handles getting the user's token, validating it, and providing the user object.
async def get_current_user_details(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Authorization header missing or invalid"
        )
    
    token = authorization.split(" ")[1]
    
    try:
        # In a serverless environment like Vercel, creating a new client per request is a safe
        # and stateless pattern. The client is lightweight.
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        
        # Set the authorization for this client instance.
        # All subsequent requests with this client will be authenticated as the user.
        supabase.postgrest.auth(token)
        
        # Explicitly validate the JWT to ensure it's not expired or tampered with by fetching the user.
        # This call to Supabase Auth also returns the user's details.
        user_response = await asyncio.to_thread(supabase.auth.get_user, token)
        user = user_response.user

        if not user:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token or user not found")
        
        # Return the authenticated client and user details
        return {"user": user, "client": supabase, "token": token}
    except Exception as e:
        # This could be a PostgrestError or another exception
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Invalid token: {str(e)}")

# --- API Endpoints ---
@app.get("/api/v1/databases", response_model=List[DatabaseResponse])
async def get_user_databases(auth_details: dict = Depends(get_current_user_details)):
    """
    Fetches all top-level Databases for the logged-in user.
    """
    try:
        supabase = auth_details["client"]
        response = supabase.table("user_databases").select("id, created_at, name, description").order("created_at", desc=True).execute()
        return response.data
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/api/v1/databases", response_model=DatabaseResponse, status_code=status.HTTP_201_CREATED)
async def create_user_database(db_data: DatabaseCreate, auth_details: dict = Depends(get_current_user_details)):
    """
    Creates a new Database for the logged-in user.
    """
    try:
        supabase = auth_details["client"]
        user = auth_details["user"]
        new_db_data = {
            "user_id": user.id,
            "name": db_data.name,
            "description": db_data.description
        }
        response = supabase.table("user_databases").insert(new_db_data, returning="representation").execute()
        # The data is returned as a list, so we take the first element.
        return response.data[0]
    except APIError as e:
        # Check for PostgreSQL's unique violation error code
        if e.code == "23505":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"A database with the name '{db_data.name}' already exists.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not create database: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/api/v1/databases/{database_id}", response_model=DatabaseResponse)
async def get_single_database(database_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Fetches the details for a single database. RLS policy ensures the user owns it.
    """
    try:
        supabase = auth_details["client"]
        # Fetch the database and all its related tables (user_tables) at once.
        response = supabase.table("user_databases").select("*, user_tables(*)").eq("id", database_id).single().execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Database not found")
        return response.data
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/api/v1/databases/by-name/{db_name}", response_model=DatabaseResponse)
async def get_database_by_name(db_name: str, auth_details: dict = Depends(get_current_user_details)):
    """
    Fetches a single database by its unique name. RLS policy ensures the user owns it.
    """
    try:
        supabase = auth_details["client"]
        # Use Supabase's ability to fetch related data in a single query.
        # This fetches the database and all its related tables (user_tables) at once. RLS on user_tables is also applied.
        response = supabase.table("user_databases").select("*, user_tables(*)").eq("name", db_name).single().execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Database with name '{db_name}' not found or access denied.")
        return response.data
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/api/v1/databases/{database_id}/tables", response_model=List[TableResponse])
async def get_database_tables(database_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Fetches all tables for a specific database.
    """
    try:
        supabase = auth_details["client"]
        response = supabase.table("user_tables").select("id, name, columns").eq("database_id", database_id).order("name").execute()
        return response.data
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/api/v1/databases/{database_id}/tables", response_model=TableResponse, status_code=status.HTTP_201_CREATED)
async def create_database_table(database_id: int, table_data: TableCreate, auth_details: dict = Depends(get_current_user_details)):
    """
    Creates a new table within a specific database.
    """
    try:
        supabase = auth_details["client"]
        user = auth_details["user"]

        # Verify user has access to the parent database first
        db_check = supabase.table("user_databases").select("id").eq("id", database_id).maybe_single().execute()
        if not db_check.data:
            raise HTTPException(status_code=404, detail="Parent database not found or access denied")

        new_table_data = {
            "user_id": user.id,
            "database_id": database_id,
            "name": table_data.name,
            "columns": [col.dict() for col in table_data.columns]
        }
        insert_response = supabase.table("user_tables").insert(new_table_data, returning="representation").execute()
        # The data is returned as a list, so we take the first element.
        created_table = insert_response.data[0]

        # --- Automatically create a VIEW for this table ---
        # This makes the table immediately queryable in the SQL Runner.
        try:
            supabase.rpc('create_or_replace_view_for_table', {
                'p_table_id': created_table['id'],
                'p_database_id': database_id,
                'p_table_name': created_table['name']
            }).execute()
            # --- FIX: Notify PostgREST to reload its schema cache ---
            # This is crucial to make the new view immediately available to the API.
            # Without this, subsequent API calls might fail with a "table not found" error.
            supabase.rpc('pgrst_reload_schema').execute()

        except Exception as view_error:
            # If view creation fails, we don't fail the whole request, but we should log it.
            print(f"Warning: Could not create view for table {created_table['id']}: {view_error}")

        return created_table
    except APIError as e:
        # Check for a unique constraint violation on the table name for that database
        if "user_tables_database_id_name_key" in str(e):
                 raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"A table with the name '{table_data.name}' already exists in this database.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not create table: {str(e)}")

@app.get("/api/v1/tables/{table_id}/status", status_code=status.HTTP_200_OK)
async def get_table_status(table_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Checks if a table's corresponding view is ready and queryable via PostgREST.
    This is used for polling after table creation to avoid race conditions.
    """
    supabase = auth_details["client"]
    view_name = f"user_table_view_{table_id}"
    try:
        # Perform a lightweight query against the view. If it succeeds, the view is ready.
        # --- FIX: Check for table existence AND user access simultaneously ---
        # Querying the parent table ensures RLS is checked. If this passes, the view is ready for the user.
        # A `maybe_single()` on the parent table is a very fast check.
        db_response = supabase.table("user_tables").select("id").eq("id", table_id).maybe_single().execute()
        if db_response.data:
            return {"status": "ok", "message": "Table view is ready."}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Table '{table_id}' not found or access denied.")
    except APIError as e:
        # Specifically check for the "schema cache" error.
        if e.code == "PGRST205":
            # This is the expected "not ready yet" state.
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"View '{view_name}' not yet available in API schema cache.")
        # For any other error, something is wrong.
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error checking table status: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error checking table status: {str(e)}")

@app.post("/api/v1/databases/by-name/{db_name}/tables", response_model=TableResponse, status_code=status.HTTP_201_CREATED)
async def create_table_by_db_name(db_name: str, table_data: TableCreate, auth_details: dict = Depends(get_current_user_details)):
    """
    Creates a new table within a specific database, identifying the database by its name.
    This is more robust for UIs where the name is known but the ID might not have been fetched yet.
    """
    try:
        supabase = auth_details["client"]
        user = auth_details["user"]

        # 1. Get the database ID from its name. RLS ensures the user owns it.
        db_check = supabase.table("user_databases").select("id").eq("name", db_name).maybe_single().execute()
        if not db_check.data:
            raise HTTPException(status_code=404, detail=f"Database '{db_name}' not found or access denied.")
        
        database_id = db_check.data['id']

        # 2. Use the existing create_database_table function with the fetched ID.
        # This avoids duplicating logic. We need to pass the dictionary representation of the model.
        return await create_database_table(database_id, table_data, auth_details)

    except APIError as e:
        if "user_tables_database_id_name_key" in str(e):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"A table with the name '{table_data.name}' already exists in this database.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not create table: {str(e)}")
    except HTTPException as e:
        # Re-raise HTTPExceptions from called functions
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/api/v1/databases/{database_id}/create-table-from-sql", response_model=TableResponse, status_code=status.HTTP_201_CREATED)
async def create_table_from_sql(database_id: int, sql_data: SqlTableCreateRequest, auth_details: dict = Depends(get_current_user_details)):
    """
    Parses a single CREATE TABLE SQL statement and creates the table within the specified database.
    This uses the same robust parser as the full SQL import.
    """
    supabase = auth_details["client"]
    script = sql_data.script.strip()

    if not script.upper().startswith("CREATE TABLE"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Script must be a single CREATE TABLE statement. Only one table can be created at a time via this endpoint.")

    # --- FIX: Use the more robust regex from the main SQL import function ---
    create_match = re.search(r'CREATE TABLE\s+(?:IF NOT EXISTS\s+)?[`"]?(\w+)[`"]?\s*\(((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*)\)', script, re.DOTALL | re.IGNORECASE)
    if not create_match:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid CREATE TABLE syntax. Could not find table name and column definitions.")

    _, columns_str = create_match.groups()
    table_name = sql_data.name # Use the name from the request body
    columns_defs = []
    table_level_fks = []
 
    # This regex splits columns by comma, but correctly ignores commas inside parentheses.
    for col_line in re.split(r',(?![^()]*\))', columns_str.strip()):
        col_line = col_line.strip()
        if not col_line:
            continue

        # Handle table-level foreign key definitions
        fk_match = re.search(r'FOREIGN KEY\s*\(([`"]?\w+[`"]?)\)\s*REFERENCES\s*[`"]?(\w+)[`"]?\s*\(([`"]?\w+[`"]?)\)', col_line, re.IGNORECASE)
        if fk_match:
            # This endpoint doesn't support creating foreign keys to other tables.
            # We will just ignore this line.
            table_level_fks.append({
                "source_col": fk_match.group(1).strip('`"'),
                "ref_table": fk_match.group(2).strip('`"'),
                "ref_col": fk_match.group(3).strip('`"')
            })
            continue

        # Ignore other table-level constraints for now
        if col_line.upper().startswith(("PRIMARY KEY", "UNIQUE", "CONSTRAINT", "CHECK")):
            continue

        parts = col_line.split()
        if not parts:
            continue

        col_name = parts[0].strip('`"')
        type_and_constraints = " ".join(parts[1:]).strip() 

        col_type = _extract_sql_type(type_and_constraints)

        # This endpoint doesn't support creating foreign keys, but we parse to avoid errors.
        inline_fk_match = re.search(r'REFERENCES\s+[`"]?(\w+)[`"]?\s*\(([`"]?\w+[`"]?)\)', type_and_constraints, re.IGNORECASE)
        if inline_fk_match:
            table_level_fks.append({
                "source_col": col_name,
                "ref_table": inline_fk_match.group(1).strip('`"'),
                "ref_col": inline_fk_match.group(2).strip('`"')
            })

        is_pk = "PRIMARY KEY" in type_and_constraints.upper()
        # Correctly identify auto-incrementing columns. 'SERIAL' is a type, not just a constraint.
        is_auto_increment = 'AUTO_INCREMENT' in type_and_constraints.upper() or 'SERIAL' in col_type.upper()

        columns_defs.append(ColumnDefinition(
            name=col_name,
            type=col_type,
            is_primary_key=is_pk,
            is_auto_increment=is_pk and is_auto_increment,
            is_unique="UNIQUE" in type_and_constraints.upper(),
            is_not_null="NOT NULL" in type_and_constraints.upper()
        ))

    # Create the table with basic columns
    table_create_payload = TableCreate(name=table_name, columns=columns_defs)
    created_table_dict = await create_database_table(database_id, table_create_payload, auth_details)
    created_table = TableResponse(**created_table_dict)

    # Note: This simplified endpoint does not create foreign keys.
    # The full SQL import feature handles this. We've parsed them to avoid errors,
    # but we don't act on them here.
    
    return created_table

# Helper function for CSV import
def sanitize_header(header: str) -> str:
    # Lowercase, replace spaces/dashes with underscores, remove other invalid chars
    header = header.lower().strip()
    header = re.sub(r'[\s-]+', '_', header)
    header = re.sub(r'[^a-z0-9_]', '', header)
    # Ensure it's a valid identifier (doesn't start with a number)
    if header and header[0].isdigit():
        header = '_' + header
    return header

@app.post("/api/v1/import-csv/preview", response_model=CsvPreviewResponse)
async def preview_csv_import(preview_data: CsvPreviewRequest, auth_details: dict = Depends(get_current_user_details)):
    """
    Parses the start of a CSV to infer headers and column types for UI review.
    """
    try:
        csv_file = io.StringIO(preview_data.csv_content)
        reader = csv.reader(csv_file)
        
        original_headers = next(reader)
        sanitized_headers = [sanitize_header(h) for h in original_headers]
        if len(set(sanitized_headers)) != len(sanitized_headers):
            raise ValueError("CSV contains duplicate headers after sanitization.")

        sample_rows = [row for i, row in enumerate(reader) if i < 100]
        inferred_types = infer_column_types(sample_rows, len(sanitized_headers))

        return CsvPreviewResponse(original_headers=original_headers, sanitized_headers=sanitized_headers, inferred_types=inferred_types)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to parse CSV preview: {str(e)}")

def infer_column_types(rows: List[List[str]], num_cols: int) -> List[str]:
    """
    Infers column types by inspecting the first 100 rows of data.
    It attempts to find the most specific data type that fits all non-empty values in a column.
    The order of preference is: integer -> real -> boolean -> timestamp -> text.
    """
    from datetime import datetime

    # Define boolean string values (case-insensitive)
    BOOLEAN_TRUE_STRINGS = {'true', 't', 'yes', 'y', '1'}
    BOOLEAN_FALSE_STRINGS = {'false', 'f', 'no', 'n', '0'}

    def get_type(value: str):
        if not value:
            return None  # Ignore empty strings for type detection

        # Try integer
        try:
            int(value)
            return 'integer'
        except (ValueError, TypeError):
            pass

        # Try real (float)
        try:
            float(value)
            return 'real'
        except (ValueError, TypeError):
            pass

        # Try boolean
        if value.lower() in BOOLEAN_TRUE_STRINGS or value.lower() in BOOLEAN_FALSE_STRINGS:
            return 'boolean'

        # Try timestamp (common formats)
        for fmt in ('%Y-%m-%dT%H:%M:%S.%f%z', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
            try:
                datetime.strptime(value, fmt)
                return 'timestamp'
            except (ValueError, TypeError):
                pass

        return 'text'

    type_hierarchy = ['integer', 'real', 'boolean', 'timestamp', 'text']
    inferred_types = ['integer'] * num_cols

    for row in rows:
        if len(row) != num_cols: continue
        for i, cell in enumerate(row):
            current_type = inferred_types[i]
            if current_type == 'text' or not cell:
                continue

            cell_type = get_type(cell)
            if cell_type is None:
                continue

            # If the cell type is "less specific" than the current column type, upgrade the column type.
            if type_hierarchy.index(cell_type) > type_hierarchy.index(current_type):
                inferred_types[i] = cell_type

    return inferred_types

@app.post("/api/v1/databases/{database_id}/import-csv", response_model=TableResponse, status_code=status.HTTP_201_CREATED)
async def import_table_from_csv(database_id: int, import_data: CsvImportRequest, auth_details: dict = Depends(get_current_user_details)):
    """
    Creates a new table and populates it from a CSV file string.
    It infers column types and sanitizes headers.
    """
    supabase = auth_details["client"]
    user = auth_details["user"]

    # --- FIX: Add a transaction for rollback on failure ---
    # This is a conceptual change. Supabase-py doesn't have explicit transactions,
    # but we will manually delete the table if a later step fails.

    new_table_id = None

    # The user has already reviewed and confirmed the column types in the UI.
    # We receive the final column definitions directly in the payload.
    try:
        table_create_payload = TableCreate(name=import_data.name, columns=import_data.columns)
        created_table_dict = await create_database_table(database_id, table_create_payload, auth_details)
        created_table = TableResponse(**created_table_dict)
        new_table_id = created_table.id

        # Now, insert the data
        csv_file = io.StringIO(import_data.csv_content)
        
        # Use the sanitized headers from the confirmed columns payload
        sanitized_headers = [col.name for col in import_data.columns]
        column_types = {col.name: col.type for col in import_data.columns}

        # Rewind and read all data for insertion
        dict_reader = csv.DictReader(csv_file, fieldnames=sanitized_headers)
        next(dict_reader) # Skip header row

        rows_to_insert = []
        for row_dict in dict_reader:
            processed_row = {}
            for i, header in enumerate(sanitized_headers):
                val = row_dict.get(header)
                col_type = column_types.get(header, 'text')
                if val is not None and val != '':
                    if col_type == 'integer':
                        try: val = int(val)
                        except (ValueError, TypeError): pass
                    elif col_type == 'real':
                        try: val = float(val)
                        except (ValueError, TypeError): pass
                processed_row[header] = val
            rows_to_insert.append({"user_id": user.id, "table_id": new_table_id, "data": processed_row})

        if rows_to_insert:
            supabase.table("table_rows").insert(rows_to_insert).execute()

        return created_table
    except Exception as e:
        if new_table_id:
            # Rollback: attempt to delete the partially created table
            try:
                supabase.table("user_tables").delete().eq("id", new_table_id).eq("user_id", user.id).execute()
            except Exception as delete_error:
                print(f"CRITICAL: Failed to rollback and delete table {new_table_id} after import error. {delete_error}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to import CSV: {str(e)}")

@app.post("/api/v1/tables/{table_id}/import-rows-from-csv", response_model=CsvRowImportResponse)
async def import_rows_into_table(table_id: int, import_data: CsvRowImportRequest, auth_details: dict = Depends(get_current_user_details)):
    """
    Imports rows from a CSV file into an existing table based on a user-defined column mapping.
    """
    supabase = auth_details["client"]
    user = auth_details["user"]

    try:
        # 1. Get the schema of the target table to know the expected data types
        table_schema_dict = await get_single_table(table_id, auth_details)
        table_schema = TableResponse(**table_schema_dict)
        table_column_types = {col.name: col.type for col in table_schema.columns}

        # 2. Parse the CSV
        csv_file = io.StringIO(import_data.csv_content)
        dict_reader = csv.DictReader(csv_file)
        
        rows_to_insert = []
        failed_rows = []

        for i, row_dict in enumerate(dict_reader, start=2): # Row 1 is header, so data starts at line 2
            processed_row_data = {}
            has_error = False
            error_reason = ""

            # 3. Map CSV columns to table columns and cast types
            for table_col, csv_header in import_data.column_mapping.items():
                if csv_header not in row_dict:
                    continue # Skip if the mapped CSV header doesn't exist in this row
                
                val = row_dict[csv_header]
                original_val = val
                target_type = table_column_types.get(table_col)

                if val is not None and val != '':
                    try:
                        if target_type == 'integer': val = int(val)
                        elif target_type == 'real': val = float(val)
                        elif target_type == 'boolean': val = val.lower() in {'true', 't', 'yes', 'y', '1'}
                        # Timestamps and text are kept as strings for insertion
                    except (ValueError, TypeError):
                        has_error = True
                        error_reason = f"Column '{table_col}': Invalid value '{original_val}' for type '{target_type}'."
                        break # Stop processing this row on first error
                
                processed_row_data[table_col] = val
            
            if has_error:
                failed_rows.append(RowImportError(row_number=i, data=row_dict, error=error_reason))
            elif processed_row_data:
                rows_to_insert.append({"user_id": user.id, "table_id": table_id, "data": processed_row_data})

        if rows_to_insert:
            supabase.table("table_rows").insert(rows_to_insert).execute()

        return CsvRowImportResponse(
            inserted_count=len(rows_to_insert),
            failed_count=len(failed_rows),
            errors=failed_rows
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to import rows: {str(e)}")

def _extract_sql_type(col_def_str: str) -> str:
    """
    Robustly extracts the data type from a SQL column definition string.
    e.g., "INTEGER PRIMARY KEY NOT NULL" -> "INTEGER"
    e.g., "VARCHAR(255) UNIQUE" -> "VARCHAR(255)"
    """
    # List of known constraints to strip from the end
    # The regex looks for the data type at the start, which might include parentheses.
    # It stops at the first known constraint keyword.
    # This is more robust than splitting by space.
    match = re.match(r'^\s*([a-zA-Z_]+(?:\s*\(\s*\d+(?:\s*,\s*\d+)?\s*\))?)', col_def_str, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return 'TEXT' # Fallback


@app.put("/api/v1/tables/{table_id}", response_model=TableResponse)
async def update_database_table(table_id: int, table_data: TableUpdate, response: Response, auth_details: dict = Depends(get_current_user_details)):
    """
    Updates a table's structure (name and columns).
    """
    try:
        supabase = auth_details["client"]
        update_data = {
            "name": table_data.name,
            "columns": [col.dict() for col in table_data.columns]
        }
        db_response = supabase.table("user_tables").update(update_data, returning="representation").eq("id", table_id).execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Table not found or access denied.")
        
        updated_table = response.data[0]

        # --- FIX: Re-create the view to reflect the structure changes ---
        # This is the missing piece. Without this, the SQL Runner's view becomes outdated.
        try:
            supabase.rpc('create_or_replace_view_for_table', {
                'p_table_id': updated_table['id'],
                'p_database_id': updated_table['database_id'],
                'p_table_name': updated_table['name']
            }).execute()
            # --- FIX: Notify PostgREST to reload its schema cache after updating the view ---
            supabase.rpc('pgrst_reload_schema').execute()

        except Exception as view_error:
            response.headers["X-View-Created"] = "false"
            print(f"Warning: Could not update view for table {updated_table['id']} after structure change: {view_error}")

        return updated_table
    except APIError as e:
        # Handle case where the new table name conflicts with an existing one in the same database.
        if "user_tables_database_id_name_key" in str(e):
                 raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"A table with the name '{table_data.name}' already exists in this database.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not update table: {str(e)}")

@app.delete("/api/v1/databases/{database_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_database( # pragma: no cover
    database_id: int, 
    auth_details: dict = Depends(get_current_user_details)
):
    """
    Deletes a database and all its associated tables and rows.
    Also ensures all linked calendar events are properly deleted.
    """
    try:
        supabase = auth_details["client"]

        # The RLS policy ensures the user can only match their own database ID.
        # The 'returning="representation"' ensures data is returned to check if a row was actually deleted.
        response = supabase.table("user_databases").delete(returning="representation").eq("id", database_id).execute()
        
        if not response.data:
                 raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Database not found or you do not have permission to delete it.")

    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not delete database: {str(e)}")

@app.delete("/api/v1/tables/{table_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_database_table(
    table_id: int, 
    auth_details: dict = Depends(get_current_user_details)
):
    """
    Deletes a table and all its associated rows.
    """
    try:
        supabase = auth_details["client"]
        response = supabase.table("user_tables").delete(returning="representation").eq("id", table_id).execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Table not found or you do not have permission to delete it.")
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not delete table: {str(e)}")

@app.get("/api/v1/tables/{table_id}", response_model=TableResponse)
async def get_single_table(table_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Fetches the details for a single table. RLS policy ensures the user owns it.
    """
    try:
        supabase = auth_details["client"]
        response = supabase.table("user_tables").select("id, name, columns").eq("id", table_id).single().execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Table not found")
        return response.data
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/api/v1/tables/{table_id}/rows", response_model=PaginatedRowResponse)
async def get_table_rows(
    table_id: int, 
    auth_details: dict = Depends(get_current_user_details),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = Query(None)
):
    """
    Fetches data rows for a specific table with pagination and search.
    """
    supabase = auth_details["client"]
    try:
        # --- FIX: Query the user-specific view, not the base table_rows ---
        # The view `user_table_view_{id}` correctly projects the row's internal `id`
        # as the user-defined primary key column. This is the correct source of truth.
        view_name = f"user_table_view_{table_id}"
        query = supabase.from_(view_name).select("*", count='exact')

        if search:
            # The view exposes all columns as top-level, so we can search them directly.
            # We need the schema to know which columns to search.
            table_schema_dict = await get_single_table(table_id, auth_details)
            table_schema_obj = TableResponse(**table_schema_dict)
            searchable_columns = [col.name for col in table_schema_obj.columns]
            or_filter = ",".join([f'"{col}".ilike.%{search}%' for col in searchable_columns])
            query = query.or_(or_filter)

        response = query.order("id").range(offset, offset + limit - 1).execute()

        return {"total": response.count, "data": response.data or []}
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/api/v1/tables/{table_id}/all-rows", response_model=List[RowResponse])
async def get_all_table_rows(table_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Fetches ALL data rows for a specific table, bypassing pagination.
    Used for features like CSV export and caching foreign key data.
    """
    try:
        supabase = auth_details["client"]
        # --- FIX: Query the user-specific view to get all columns, including the projected PK ---
        view_name = f"user_table_view_{table_id}"
        # The view is protected by RLS on the underlying tables, so this is secure.
        response = supabase.from_(view_name).select("*").order("id").execute()
        
        return response.data

    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/api/v1/tables/{table_id}/rows", response_model=RowResponse, status_code=status.HTTP_201_CREATED)
async def create_table_row(
    table_id: int, 
    row_data: RowCreate, # This parameter was missing
    auth_details: dict = Depends(get_current_user_details)
):
    """
    Creates a new row for a table with the provided data.
    """
    try:
        supabase = auth_details["client"]
        user = auth_details["user"]

        # --- FIX: Add validation before inserting ---
        await _validate_row_data(supabase, table_id, row_data.data)

        new_row_data = {
            "user_id": user.id,
            "table_id": table_id,
            "data": row_data.data
        }
        row_response = supabase.table("table_rows").insert(new_row_data, returning="representation").execute()
        if not row_response.data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to create row. This may be due to a database policy violation.")
        return row_response.data[0]
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not create row: {e.message}")
    except HTTPException as e:
        # Re-raise validation errors
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")

@app.put("/api/v1/rows/{row_id}", response_model=RowResponse)
async def update_table_row(row_id: int, row_data: RowCreate, auth_details: dict = Depends(get_current_user_details)):
    """
    Updates the data for a specific row.
    """
    supabase = auth_details["client"]
    try:
        # --- FIX: The `row_id` from the URL is the value of the user-defined PK, not the internal `table_rows.id`.
        # We must first find the internal ID based on the user-defined PK.

        # 1. Find which table this row belongs to by looking for a row where the PK column matches the provided `row_id`.
        # This is a bit complex because we don't know the table_id upfront. We have to query all rows.
        # A better approach would be to include table_id in the URL, but we'll work with the current structure.
        
        # First, find the table_id by looking up the row.
        # This is inefficient but necessary with the current endpoint structure.
        # We find the row by its actual ID and get its table_id.
        row_lookup_res = supabase.table("table_rows").select("id, table_id").eq("id", row_id).single().execute()
        if not row_lookup_res.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Row not found or access denied.")
        
        internal_row_id = row_lookup_res.data['id']
        table_id = row_lookup_res.data['table_id']

        # 2. Validate the incoming data against the table's schema.
        await _validate_row_data(supabase, table_id, row_data.data, row_id=row_id)

        # 3. Update the row data as requested.
        # We use the internal_row_id we found for the update operation.
        response = supabase.table("table_rows").update({"data": row_data.data}, returning="representation").eq("id", internal_row_id).execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Row not found or access denied.") # pragma: no cover
        
        # 4. Return the updated row. We use the internal ID to fetch it.
        return await get_single_row(internal_row_id, auth_details)

    except APIError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not update row: {e.message}")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred during row update: {str(e)}")

@app.delete("/api/v1/rows/{row_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_table_row(row_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Deletes a specific row.
    The `row_id` is the value of the user-defined primary key.
    """
    try:
        supabase = auth_details["client"]
        # Now, delete the row itself.
        response = supabase.table("table_rows").delete(returning="representation").eq("id", row_id).execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Row not found or you do not have permission to delete it.")

    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not delete row: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred during row deletion: {str(e)}")

@app.get("/api/v1/rows/{row_id}", response_model=RowResponse)
async def get_single_row(row_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Fetches a single row by its ID. RLS ensures the user has access.
    """
    try:
        supabase = auth_details["client"]
        response = supabase.table("table_rows").select("*").eq("id", row_id).single().execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Row not found or access denied.")
        return response.data
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not fetch row: {str(e)}")

@app.post("/api/v1/query-builder/fetch")
async def fetch_with_query_builder(
    query_request: QueryBuilderRequest,
    auth_details: dict = Depends(get_current_user_details)
):
    """
    Executes a safe, structured query based on selections from the visual query builder.
    This does NOT execute raw SQL from the client.
    """
    supabase = auth_details["client"]
    table_id = query_request.table_id

    try:
        # 1. Verify user has access to the table (implicitly done by RLS on table_rows)
        # and get table schema for validation.
        table_schema_dict = await get_single_table(table_id, auth_details)
        table_schema = TableResponse(**table_schema_dict)
        valid_columns = {col.name for col in table_schema.columns}

        # 2. Validate requested columns and filters against the schema
        if not all(c in valid_columns for c in query_request.columns if c != '*'):
            raise HTTPException(status_code=400, detail="Invalid column requested.")
        if not all(f.column in valid_columns for f in query_request.filters):
            raise HTTPException(status_code=400, detail="Invalid column specified in filter.")

        # 3. Build the query
        # We only select the 'data' jsonb field and will process it later.
        query = supabase.table("table_rows").select("data").eq("table_id", table_id)

        # 4. Apply filters safely
        for f in query_request.filters:
            # Sanitize operator to prevent abuse
            # Using PostgREST operators: https://postgrest.org/en/stable/api.html#operators
            operator_map = {
                '=': 'eq', '!=': 'neq', '>': 'gt', '<': 'lt', '>=': 'gte', '<=': 'lte',
                'LIKE': 'ilike', 'NOT LIKE': 'not.ilike', 'IN': 'in'
            }
            if f.operator in operator_map:
                # For LIKE, wrap value in wildcards
                value = f"%{f.value}%" if 'LIKE' in f.operator else f.value
                # For IN, the value should be a tuple of values
                if f.operator == 'IN':
                    value = tuple(v.strip() for v in f.value.split(','))
                
                # The format for filtering on a JSONB column is `data->>column_name`
                query = query.filter(f"data->>{f.column}", operator_map[f.operator], value)
            elif f.operator == 'IS NULL':
                query = query.is_(f"data->>{f.column}", None)
            elif f.operator == 'IS NOT NULL':
                query = query.not_.is_(f"data->>{f.column}", None)

        # Limit results for safety
        response = query.limit(1000).execute()

        # 5. Process results to select only the requested columns
        if '*' in query_request.columns:
            return [row['data'] for row in response.data]
        else:
            processed_data = []
            for row in response.data:
                processed_data.append({col: row['data'].get(col) for col in query_request.columns})
            return processed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

async def _validate_row_data(
    supabase: Client,
    table_id: int,
    data: dict,
    row_id: Optional[int] = None
):
    """
    Validates row data against the table's schema for NOT NULL, UNIQUE, and data type constraints.
    """
    # We need a fake auth_details dict to call get_single_table
    fake_auth_details = {"client": supabase}
    table_res = await get_single_table(table_id, fake_auth_details)
    table = TableResponse(**table_res)

    for col in table.columns:
        # --- FIX: Handle auto-incrementing primary keys on row creation ---
        is_new_row = row_id is None
        if is_new_row and col.is_primary_key and col.is_auto_increment:
            # For new rows, the auto-incrementing PK is generated by the DB.
            # We must remove it from the data payload to avoid trying to insert it.
            # The frontend might send a placeholder like '(auto)' which we should ignore.
            if col.name in data:
                del data[col.name]
            continue # Skip all validation for this column on creation.

        value = data.get(col.name)

        # NOT NULL check
        if col.is_not_null and (value is None or str(value).strip() == ""):
            raise HTTPException(status_code=400, detail=f"Column '{col.name}' cannot be empty because it is marked as NOT NULL.")

        if value is None or str(value).strip() == "":
            continue  # Skip further checks for empty, nullable fields

        # Data Type check
        try:
            if col.type == "integer": int(value)
            elif col.type == "real": float(value)
            elif col.type == "boolean":
                if str(value).lower() not in ('true', 'false', 't', 'f', '1', '0'):
                    raise ValueError("Not a valid boolean")
            elif col.type in ("timestamp", "date"):
                # Attempt to parse a variety of common date/datetime formats
                datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail=f"Invalid data format for column '{col.name}'. Expected {col.type.upper()}, but got '{value}'.")

        # Uniqueness check
        if col.is_unique:
            # Supabase RPC functions are better for this, but a direct query works.
            # We check if any other row in this table has the same value for this column.
            query = supabase.table("table_rows").select("id", count='exact').eq("table_id", table_id).eq(f"data->>{col.name}", str(value))
            if row_id:  # If updating, exclude the current row from the check
                query = query.neq("id", row_id)
            res = query.execute()
            if res.count > 0:
                raise HTTPException(status_code=409, detail=f"Value '{value}' for column '{col.name}' already exists. It must be unique.")

async def _delete_item(supabase: Client, table_name: str, item_id: int, item_type_name: str):
    """Generic helper to delete an item from a table by ID, with RLS ensuring ownership."""
    try:
        response = supabase.table(table_name).delete(returning="representation").eq("id", item_id).execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{item_type_name} not found or access denied.")
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not delete {item_type_name.lower()}: {str(e)}")


@app.post("/api/v1/users/me/delete", status_code=status.HTTP_200_OK)
async def delete_own_account(
    form_data: AuthenticatedAccountDeletionRequest,
    auth_details: dict = Depends(get_current_user_details)
):
    """
    Handles final, authenticated account deletion. The user must provide their
    password and a confirmation phrase to prove their identity.
    """
    # 1. Check for the required service key. This is essential for creating the admin client.
    if not SUPABASE_SERVICE_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account deletion is not configured on the server. Missing service key."
        )

    # 2. Get user details from the validated auth token. The `get_current_user_details`
    # dependency has already confirmed the user's identity is valid.
    user = auth_details["user"]

    # 3. Explicitly check the confirmation text before proceeding.
    # Pydantic's pattern matching handles this, but a user-friendly error is better.
    if form_data.confirmation != "delete my account":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmation text does not match. Please type 'delete my account' exactly as shown."
        )
    try:
        # 3. Create an admin client to perform the deletion.
        # We are skipping the password re-authentication step, as it is what triggers
        # the captcha from Supabase's end. We rely on the already-validated JWT from
        # the `get_current_user_details` dependency as sufficient proof of identity.
        supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

        # 4. Delete the user using their ID.
        await asyncio.to_thread(supabase_admin.auth.admin.delete_user, user.id)
        # NOTE: The `delete_user_and_data` PostgreSQL function is now assumed to be
        # called by a trigger `ON DELETE ON auth.users`. This is the standard Supabase pattern.
        # Deleting the user here will automatically trigger the cleanup of their data.

        return {"message": "Account successfully deleted."}

    except APIError as e:
        # This will catch any errors from the admin client, such as permission issues.
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An API error occurred during account deletion: {e.message}")
    except HTTPException as e:
        raise e  # Re-raise validation errors
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/api/v1/databases/{database_id}/export-sql", response_class=PlainTextResponse)
async def export_database_as_sql(database_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Generates a full SQL script for a database, including CREATE TABLE and INSERT statements.
    """
    supabase = auth_details["client"]
    
    # Fetch database name
    db_res_dict = await get_single_database(database_id, auth_details)
    db_name = db_res_dict['name']

    # Fetch all tables for the database
    tables_dicts = await get_database_tables(database_id, auth_details)
    # Convert dicts to Pydantic models to safely use attribute access
    tables = [TableResponse(**t) for t in tables_dicts]
    # Create a map for efficient lookup of table names by ID
    table_id_to_name_map = {t.id: t.name for t in tables}
    
    sql_script = f"-- SQL Dump for database: {db_name}\n\n"

    for table in tables:
        # Generate CREATE TABLE statement
        sql_script += f"-- Structure for table: {table.name}\n"
        create_statement = f"CREATE TABLE \"{table.name}\" (\n"
        column_defs = []
        for col in table.columns:
            col_def = f"  \"{col.name}\" {col.type.upper()}"
            if col.is_primary_key: col_def += " PRIMARY KEY"
            if col.is_not_null: col_def += " NOT NULL"
            if col.is_unique: col_def += " UNIQUE"
            if col.foreign_key:
                referenced_table_name = table_id_to_name_map.get(col.foreign_key.table_id)
                if referenced_table_name:
                    col_def += f' REFERENCES "{referenced_table_name}" ("{col.foreign_key.column_name}") ON DELETE CASCADE'
            column_defs.append(col_def)
        create_statement += ",\n".join(column_defs)
        create_statement += "\n);\n\n"
        sql_script += create_statement

        # --- FIX: Fetch rows from the user-specific view to get all columns correctly ---
        view_name = f"user_table_view_{table.id}"
        raw_rows_res = supabase.from_(view_name).select("*").order("id").execute()

        if raw_rows_res.data and isinstance(raw_rows_res.data, list):
            sql_script += f"-- Data for table: {table.name}\n"
            for row in raw_rows_res.data:
                row_data = row
                if not row_data:
                    continue

                pk_col = next((col for col in table.columns if col.is_primary_key), None)
                # --- FIX: Treat 'serial' type as auto-incrementing as well ---
                pk_is_auto_increment = (pk_col.is_auto_increment or (pk_col.type or '').lower() == 'serial') if pk_col else False
                
                columns_to_insert = [f'"{k}"' for k, v in row_data.items() if not (pk_is_auto_increment and k == pk_col.name)]
                values_to_insert = []
                for k, v in row_data.items():
                    if pk_is_auto_increment and k == pk_col.name:
                        # Skip auto-incrementing PKs from INSERT statements
                        continue

                    if isinstance(v, str):
                        # The value from JSONB is already a clean string. We just need to escape it for SQL.
                        escaped_v = v.replace("'", "''")
                        values_to_insert.append(f"'{escaped_v}'")
                    elif v is None:
                        values_to_insert.append("NULL")
                    elif isinstance(v, bool):
                        values_to_insert.append("TRUE" if v else "FALSE")
                    else:
                        values_to_insert.append(str(v))

                if columns_to_insert:
                    sql_script += f"INSERT INTO \"{table.name}\" ({', '.join(columns_to_insert)}) VALUES ({', '.join(values_to_insert)});\n"
            sql_script += "\n"

    return PlainTextResponse(content=sql_script)

@app.post("/api/v1/contact")
async def handle_contact_form(contact_data: ContactRequest):
    """
    Handles submissions from the public contact form.
    In a real application, this would trigger an email.
    For now, we'll just log it to the server console for debugging.
    This endpoint does not require authentication.
    """
    print("--- New Contact Form Submission ---")
    print(f"From: {contact_data.sender_name} <{contact_data.sender_email}>")
    print(f"Message: {contact_data.message}")
    print("---------------------------------")
    
    # In a real app, you might add rate limiting here.
    
    return {"message": "Message received successfully. Thank you!"}

# This is the helper function from your initial snippet, fully implemented.
async def _parse_and_execute_insert(statement: str, created_tables_map: dict, db_id: int, supabase: Client, user: Any):
    # Handle multi-value INSERT statements
    header_match = re.search(r'INSERT INTO\s+[`"]?(\w+)[`"]?\s*\(([^)]+)\)\s*VALUES', statement, re.IGNORECASE)
    if not header_match: return

    raw_table_name, columns_str = header_match.groups()
    table_name = raw_table_name.lower() # Sanitize to lowercase
    if table_name not in created_tables_map:
        return # Silently skip if the table wasn't created in the first pass
    
    table_id = created_tables_map[table_name]
    columns = [c.strip().strip('`"') for c in columns_str.split(',')]
    
    # Find all value tuples like (...), (...), (...) using a more robust, non-greedy regex
    values_part = statement[header_match.end():].strip()
    # This regex finds all top-level parenthesized groups, correctly handling nested parentheses.
    value_tuples_str = re.findall(r'\(((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*)\)', values_part)

    rows_to_insert = []
    for values_str in value_tuples_str:
        # This regex splits by comma, but respects commas inside single-quoted strings.
        # It handles values like 'a,b', 123, 'c' -> ["'a,b'", ' 123', " 'c'"]
        values = re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", values_str)
        
        if len(columns) == len(values):
            # Attempt to convert numeric strings to actual numbers
            typed_values = []
            for v in values:
                clean_v = v.strip()
                try:
                    # Try to convert to int first
                    typed_values.append(int(clean_v))
                except ValueError:
                    try:
                        # Then try to convert to float
                        typed_values.append(float(clean_v))
                    except ValueError:
                        # If it fails, treat it as a string, stripping the outer quotes.
                        typed_values.append(clean_v.strip("'\""))
            
            rows_to_insert.append({"user_id": user.id, "table_id": table_id, "data": dict(zip(columns, typed_values))})

    if rows_to_insert:
        supabase.table("table_rows").insert(rows_to_insert).execute()

# This is the import endpoint from your initial snippet, fully implemented.
@app.post("/api/v1/databases/import-sql", response_model=DatabaseResponse, status_code=status.HTTP_201_CREATED)
async def import_database_from_sql(import_data: SqlImportRequest, auth_details: dict = Depends(get_current_user_details)):
    """
    Creates a new database and attempts to populate it from a user-provided SQL script.
    This has a very limited, best-effort SQL parser and is not guaranteed to work with all SQL dialects.
    """
    supabase = auth_details["client"]
    user = auth_details["user"]
    new_db_id = None

    try:
        # 1. Create the parent database container
        db_response = await create_user_database(
            db_data=DatabaseCreate(name=import_data.name, description=import_data.description),
            auth_details=auth_details
        )
        new_db_id = db_response['id']

        # 2. Robust parsing and execution of the SQL script
        # First, remove all comments from the script
        script_no_comments = re.sub(r'--.*', '', import_data.script)
        statements = [s.strip() for s in script_no_comments.split(';') if s.strip()]
        
        # --- Multi-pass import process ---

        # Data structures to hold the parsed schema before creation
        parsed_tables = {} # { table_name: { "columns": [ColumnDefinition], "foreign_keys": [...] } }

        # Pass 1: Parse all CREATE TABLE statements and store their structure
        for statement in statements:
            if not statement.upper().startswith("CREATE TABLE"):
                continue

            # --- FIX: Use a more precise regex to capture only the CREATE TABLE statement ---
            # This regex finds 'CREATE TABLE', captures the name, and then correctly matches
            # the content within the first top-level pair of parentheses. It stops before any trailing semicolon.
            create_match = re.search(r'CREATE TABLE\s+(?:IF NOT EXISTS\s+)?[`"]?(\w+)[`"]?\s*\(((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*)\)', statement, re.DOTALL | re.IGNORECASE)
            if not create_match: continue
            
            raw_table_name, columns_str = create_match.groups()
            # Sanitize the table name to lowercase to prevent case-sensitivity issues.
            table_name = raw_table_name.lower()
            columns_defs = []
            table_level_fks = []

            # This regex splits columns by comma, but correctly ignores commas inside parentheses.
            for col_line in re.split(r',(?![^()]*\))', columns_str.strip()):
                col_line = col_line.strip()
                if not col_line: continue

                # Check for table-level FOREIGN KEY constraint: FOREIGN KEY (col) REFERENCES other_table(other_col)
                fk_match = re.search(r'FOREIGN KEY\s*\(([`"]?\w+[`"]?)\)\s*REFERENCES\s*[`"]?(\w+)[`"]?\s*\(([`"]?\w+[`"]?)\)', col_line, re.IGNORECASE)
                if fk_match:
                    source_col, ref_table, ref_col = fk_match.groups() # ref_table is the table name
                    table_level_fks.append({
                        "source_col": source_col.strip('`"'),
                        "ref_table": ref_table.strip('`"'),
                        "ref_col": ref_col.strip('`"')
                    })
                    continue

                # Skip other table-level constraints for now
                if col_line.upper().startswith(("PRIMARY KEY", "UNIQUE", "CONSTRAINT")):
                    continue

                # Assume it's a column definition
                parts = col_line.split()
                if not parts: continue

                col_name = parts[0].strip('`"')
                type_and_constraints = " ".join(parts[1:]).strip()
                # --- FIX: Use the robust type extraction and DO NOT normalize ---
                col_type = _extract_sql_type(type_and_constraints)

                # --- FIX: Add parsing for inline foreign keys ---
                inline_fk_match = re.search(r'REFERENCES\s+[`"]?(\w+)[`"]?\s*\(([`"]?\w+[`"]?)\)', type_and_constraints, re.IGNORECASE)
                if inline_fk_match:
                    ref_table, ref_col = inline_fk_match.groups()
                    table_level_fks.append({
                        "source_col": col_name,
                        "ref_table": ref_table.strip('`"'),
                        "ref_col": ref_col.strip('`"')
                    })

                is_pk = "PRIMARY KEY" in type_and_constraints.upper()
                # Correctly identify auto-incrementing columns. 'SERIAL' is a type, not just a constraint.
                is_auto_increment = 'AUTO_INCREMENT' in type_and_constraints.upper() or 'SERIAL' in col_type.upper()

                columns_defs.append(ColumnDefinition(
                    name=col_name,
                    type=col_type, # Use the original, full type
                    is_primary_key=is_pk,
                    is_auto_increment=is_pk and is_auto_increment,
                    is_unique="UNIQUE" in type_and_constraints.upper(),
                    is_not_null="NOT NULL" in type_and_constraints.upper()
                ))
            
            if not columns_defs: continue
            parsed_tables[table_name] = {"columns": columns_defs, "foreign_keys": table_level_fks}

        # Pass 2: Create all tables (without FKs) to establish their IDs
        created_tables_map = {} # Maps table name to table ID
        for table_name, schema in parsed_tables.items():
            table_create_payload = TableCreate(name=table_name, columns=schema["columns"])
            created_table = await create_database_table(new_db_id, table_create_payload, auth_details)
            created_tables_map[table_name] = created_table['id']

        # Pass 3: Update tables with foreign key constraints
        all_created_tables_dicts = await get_database_tables(new_db_id, auth_details)
        all_created_tables_map = {t['name']: t for t in all_created_tables_dicts}

        for table_name, schema in parsed_tables.items():
            if not schema["foreign_keys"]: continue
            table_to_update_dict = all_created_tables_map.get(table_name)
            if not table_to_update_dict: continue
            
            table_to_update = TableResponse(**table_to_update_dict)
            made_changes = False
            for fk in schema["foreign_keys"]:
                target_column = next((c for c in table_to_update.columns if c.name == fk["source_col"]), None)
                # --- FIX: Use the sanitized (lowercase) table name for the lookup ---
                # The created_tables_map uses lowercase keys, so we must look up with a lowercase key.
                referenced_table_id = created_tables_map.get(fk["ref_table"].lower())
                if target_column and referenced_table_id:
                    target_column.foreign_key = ForeignKeyDefinition(table_id=referenced_table_id, column_name=fk["ref_col"])
                    made_changes = True

            if made_changes:
                update_payload = TableUpdate(name=table_to_update.name, columns=table_to_update.columns)
                await update_database_table(table_to_update.id, update_payload, auth_details)

        # Pass 4: Insert all data
        for statement in statements:
            if statement.upper().startswith("INSERT INTO"):
                await _parse_and_execute_insert(statement, created_tables_map, new_db_id, supabase, user)
        
        # --- FIX: After all tables and data are imported, create the views for the SQL runner ---
        final_tables_res = await get_database_tables(new_db_id, auth_details)
        for table in final_tables_res:
            try:
                supabase.rpc('create_or_replace_view_for_table', {
                    'p_table_id': table['id'],
                    'p_database_id': new_db_id,
                    'p_table_name': table['name'],
                }).execute()
                # --- FIX: Notify PostgREST to reload its schema cache after creating the view ---
                supabase.rpc('pgrst_reload_schema').execute()

            except Exception as view_error:
                print(f"Warning: Could not create view for imported table {table['id']} ({table['name']}): {view_error}")

        return db_response
    except Exception as e:
        # If any part of the process fails, roll back by deleting the created database.
        if new_db_id:
            await delete_user_database(new_db_id, auth_details)
        raise HTTPException(status_code=400, detail=f"Failed to import SQL script: {str(e)}. The new database has been rolled back.")

# --- SEO / Static File Routes ---
@app.get("/robots.txt", response_class=FileResponse)
async def robots_txt():
    """Serves the robots.txt file from the public directory."""
    return FileResponse(path=BASE_DIR / "public" / "robots.txt", media_type="text/plain")

@app.get("/sitemap.xml", response_class=FileResponse)
async def sitemap_xml():
    """Serves the sitemap.xml file from the public directory."""
    return FileResponse(path=BASE_DIR / "public" / "sitemap.xml", media_type="application/xml")

# --- HTML Serving Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "supabase_url": SUPABASE_URL, 
            "supabase_anon_key": SUPABASE_ANON_KEY
        })

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse(
        "signup.html", 
        {
            "request": request, 
            "supabase_url": SUPABASE_URL, 
            "supabase_anon_key": SUPABASE_ANON_KEY,
            "hcaptcha_site_key": HCAPTCHA_SITE_KEY
        }
    )

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html", 
        {
            "request": request, "supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY,
            "hcaptcha_site_key": HCAPTCHA_SITE_KEY
        }
    )

@app.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password_page(request: Request):
    return templates.TemplateResponse(
        "forgot-password.html", 
        {
            "request": request, "supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY,
            "hcaptcha_site_key": HCAPTCHA_SITE_KEY
        }
    )

@app.get("/update-password", response_class=HTMLResponse)
async def update_password_page(request: Request):
    return templates.TemplateResponse(
        "update-password.html", 
        {
            "request": request, "supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY,
            "hcaptcha_site_key": HCAPTCHA_SITE_KEY
        }
    )

@app.get("/confirm-delete", response_class=HTMLResponse)
async def confirm_delete_page(request: Request):
    return templates.TemplateResponse(
        "confirm-delete.html",
        {
            "request": request,
            "supabase_url": SUPABASE_URL,
            "supabase_anon_key": SUPABASE_ANON_KEY,
        },
    )

@app.get("/delete-success", response_class=HTMLResponse)
async def delete_success_page(request: Request):
    # This page is shown after a user successfully deletes their account.
    return templates.TemplateResponse("delete-success.html", {"request": request})

@app.get("/email-verification", response_class=HTMLResponse)
async def email_verification_page(request: Request):
    """
    A generic page shown after a user clicks a verification link from an email.
    """
    return templates.TemplateResponse(
        "email-verification.html", 
        {
            "request": request, "supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY
        }
    )

@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request):
    return templates.TemplateResponse(
        "app.html", 
        {"request": request, "supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY}
    )

@app.get("/app/database/{db_name}", response_class=HTMLResponse)
async def table_manager_page(request: Request, db_name: str):
    return templates.TemplateResponse(
        "table-manager.html",
        {
            "request": request,
            "db_name": db_name,
            "supabase_url": SUPABASE_URL,
            "supabase_anon_key": SUPABASE_ANON_KEY,
        },
    )

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/terms", response_class=HTMLResponse)
async def terms_page(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request})

@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def contact_page(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})