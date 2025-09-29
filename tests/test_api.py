import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, ANY

# Assume your FastAPI app is in `main.py` and is named `app`
# from main import app

# Since we don't have the real app, we'll mock it for demonstration.
# In a real scenario, you would import your actual FastAPI app.
from fastapi import FastAPI, Depends, HTTPException, status

app = FastAPI()

# Mock dependency for getting current user
async def mock_get_current_user():
    return {"id": "user-uuid-123", "email": "test@example.com"}

# This is a placeholder for your actual API endpoints.
# The tests will patch the service functions called by these endpoints.
@app.get("/api/v1/databases")
async def get_databases(current_user: dict = Depends(mock_get_current_user)):
    pass

@app.post("/api/v1/databases")
async def create_database(current_user: dict = Depends(mock_get_current_user)):
    pass

@app.delete("/api/v1/databases/{db_id}")
async def delete_database(db_id: int, current_user: dict = Depends(mock_get_current_user)):
    pass

@app.get("/api/v1/databases/by-name/{db_name}")
async def get_database_by_name(db_name: str, current_user: dict = Depends(mock_get_current_user)):
    pass

@app.post("/api/v1/databases/by-name/{db_name}/tables")
async def create_table(db_name: str, current_user: dict = Depends(mock_get_current_user)):
    pass

@app.delete("/api/v1/tables/{table_id}")
async def delete_table(table_id: int, current_user: dict = Depends(mock_get_current_user)):
    pass

@app.get("/api/v1/tables/{table_id}/rows")
async def get_rows(table_id: int, current_user: dict = Depends(mock_get_current_user)):
    pass

@app.post("/api/v1/tables/{table_id}/rows")
async def create_row(table_id: int, current_user: dict = Depends(mock_get_current_user)):
    pass

@app.put("/api/v1/rows/{row_id}")
async def update_row(row_id: int, current_user: dict = Depends(mock_get_current_user)):
    pass

@app.delete("/api/v1/rows/{row_id}")
async def delete_row(row_id: int, current_user: dict = Depends(mock_get_current_user)):
    pass

@app.post("/api/v1/users/me/delete")
async def delete_user_account(current_user: dict = Depends(mock_get_current_user)):
    pass

@app.post("/api/v1/contact")
async def contact_form():
    pass


client = TestClient(app)


@pytest.fixture
def auth_headers():
    """Provides mock authorization headers for test requests."""
    return {"Authorization": "Bearer fake-token"}


@pytest.fixture(autouse=True)
def mock_auth_dependency():
    """
    Automatically mock the `get_current_user` dependency for all tests
    to simulate an authenticated user.
    """
    app.dependency_overrides[mock_get_current_user] = lambda: {
        "id": "user-uuid-123",
        "email": "test@example.com"
    }
    yield
    app.dependency_overrides = {}


# ===================================
# User Management Tests
# ===================================

@patch("main.user_service.delete_user")
def test_delete_user_account_success(mock_delete_user, auth_headers):
    """Test successful deletion of a user account."""
    mock_delete_user.return_value = True

    response = client.post(
        "/api/v1/users/me/delete",
        headers=auth_headers,
        json={"confirmation": "delete my account"}
    )

    assert response.status_code == 200
    assert response.json() == {"message": "Account deleted successfully"}
    mock_delete_user.assert_called_once_with(user_id="user-uuid-123")


def test_delete_user_account_invalid_confirmation(auth_headers):
    """Test account deletion with incorrect confirmation text."""
    response = client.post(
        "/api/v1/users/me/delete",
        headers=auth_headers,
        json={"confirmation": "wrong text"}
    )
    # This validation is likely in the Pydantic model or endpoint logic
    # Assuming it returns a 422 Unprocessable Entity
    assert response.status_code == 422


# ===================================
# Contact Form Tests
# ===================================

@patch("main.notification_service.send_contact_email")
def test_contact_form_success(mock_send_email):
    """Test successful submission of the contact form."""
    contact_data = {
        "sender_name": "John Doe",
        "sender_email": "john.doe@example.com",
        "message": "This is a test message."
    }
    response = client.post("/api/v1/contact", json=contact_data)

    assert response.status_code == 200
    assert response.json() == {"message": "Message sent successfully"}
    mock_send_email.assert_called_once_with(
        sender_name="John Doe",
        sender_email="john.doe@example.com",
        message="This is a test message."
    )


def test_contact_form_invalid_data():
    """Test contact form submission with missing fields."""
    response = client.post("/api/v1/contact", json={"sender_name": "John Doe"})
    assert response.status_code == 422  # Unprocessable Entity


# ===================================
# Database Management Tests
# ===================================

@patch("main.db_service.get_user_databases")
def test_get_databases_success(mock_get_user_databases, auth_headers):
    """Test fetching all databases for a user."""
    mock_dbs = [
        {"id": 1, "name": "db1", "description": "First DB"},
        {"id": 2, "name": "db2", "description": "Second DB"},
    ]
    mock_get_user_databases.return_value = mock_dbs

    response = client.get("/api/v1/databases", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == mock_dbs
    mock_get_user_databases.assert_called_once_with(user_id="user-uuid-123")


@patch("main.db_service.create_database")
def test_create_database_success(mock_create_db, auth_headers):
    """Test successful creation of a new database."""
    new_db_data = {"name": "new_db", "description": "A new test database"}
    created_db = {"id": 3, **new_db_data}
    mock_create_db.return_value = created_db

    response = client.post("/api/v1/databases", headers=auth_headers, json=new_db_data)

    assert response.status_code == 201
    assert response.json() == created_db
    mock_create_db.assert_called_once_with(
        user_id="user-uuid-123",
        name="new_db",
        description="A new test database"
    )


@patch("main.db_service.create_database")
def test_create_database_duplicate_name(mock_create_db, auth_headers):
    """Test creating a database with a name that already exists."""
    mock_create_db.side_effect = HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Database with this name already exists."
    )

    response = client.post(
        "/api/v1/databases",
        headers=auth_headers,
        json={"name": "existing_db", "description": ""}
    )

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


@patch("main.db_service.delete_database")
def test_delete_database_success(mock_delete_db, auth_headers):
    """Test successful deletion of a database."""
    db_id_to_delete = 1
    mock_delete_db.return_value = True

    response = client.delete(f"/api/v1/databases/{db_id_to_delete}", headers=auth_headers)

    assert response.status_code == 204
    mock_delete_db.assert_called_once_with(user_id="user-uuid-123", db_id=db_id_to_delete)


@patch("main.db_service.delete_database")
def test_delete_database_not_found(mock_delete_db, auth_headers):
    """Test deleting a database that does not exist or user does not own."""
    mock_delete_db.side_effect = HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Database not found."
    )

    response = client.delete("/api/v1/databases/999", headers=auth_headers)

    assert response.status_code == 404
    assert response.json()["detail"] == "Database not found."


# ===================================
# Table Management Tests
# ===================================

@patch("main.db_service.get_database_by_name_with_tables")
def test_get_database_by_name(mock_get_db, auth_headers):
    """Test fetching a single database with its tables by name."""
    db_name = "my_project_db"
    mock_db_with_tables = {
        "id": 1,
        "name": db_name,
        "user_tables": [{"id": 101, "name": "users"}, {"id": 102, "name": "products"}]
    }
    mock_get_db.return_value = mock_db_with_tables

    response = client.get(f"/api/v1/databases/by-name/{db_name}", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == mock_db_with_tables
    mock_get_db.assert_called_once_with(user_id="user-uuid-123", db_name=db_name)


@patch("main.table_service.create_table")
def test_create_table_success(mock_create_table, auth_headers):
    """Test successful creation of a new table."""
    db_name = "my_project_db"
    table_data = {
        "name": "new_table",
        "columns": [
            {"name": "id", "type": "SERIAL", "is_primary_key": True},
            {"name": "name", "type": "TEXT", "is_not_null": True},
        ]
    }
    mock_created_table = {"id": 103, **table_data}
    mock_create_table.return_value = mock_created_table

    response = client.post(
        f"/api/v1/databases/by-name/{db_name}/tables",
        headers=auth_headers,
        json=table_data
    )

    assert response.status_code == 201
    assert response.json() == mock_created_table
    mock_create_table.assert_called_once_with(
        user_id="user-uuid-123",
        db_name=db_name,
        table_data=ANY  # Can assert specific fields of the Pydantic model here
    )


@patch("main.table_service.delete_table")
def test_delete_table_success(mock_delete_table, auth_headers):
    """Test successful deletion of a table."""
    table_id_to_delete = 101
    mock_delete_table.return_value = True

    response = client.delete(f"/api/v1/tables/{table_id_to_delete}", headers=auth_headers)

    assert response.status_code == 204
    mock_delete_table.assert_called_once_with(user_id="user-uuid-123", table_id=table_id_to_delete)


@patch("main.table_service.delete_table")
def test_delete_table_not_found(mock_delete_table, auth_headers):
    """Test deleting a table that doesn't exist or isn't owned by the user."""
    mock_delete_table.side_effect = HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Table not found."
    )

    response = client.delete("/api/v1/tables/9999", headers=auth_headers)

    assert response.status_code == 404


# ===================================
# Row Management Tests
# ===================================

@patch("main.row_service.get_rows_paginated")
def test_get_rows_success(mock_get_rows, auth_headers):
    """Test fetching rows from a table with pagination."""
    table_id = 101
    mock_rows_response = {
        "total": 100,
        "data": [
            {"id": 1, "data": {"name": "Alice", "email": "alice@example.com"}},
            {"id": 2, "data": {"name": "Bob", "email": "bob@example.com"}},
        ]
    }
    mock_get_rows.return_value = mock_rows_response

    response = client.get(f"/api/v1/tables/{table_id}/rows?limit=2&offset=0", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == mock_rows_response
    mock_get_rows.assert_called_once_with(
        user_id="user-uuid-123",
        table_id=table_id,
        limit=2,
        offset=0,
        search_term=None
    )


@patch("main.row_service.get_rows_paginated")
def test_get_rows_with_search(mock_get_rows, auth_headers):
    """Test fetching rows with a search term."""
    table_id = 101
    search_term = "alice"
    mock_rows_response = {
        "total": 1,
        "data": [{"id": 1, "data": {"name": "Alice", "email": "alice@example.com"}}]
    }
    mock_get_rows.return_value = mock_rows_response

    response = client.get(
        f"/api/v1/tables/{table_id}/rows?search={search_term}",
        headers=auth_headers
    )

    assert response.status_code == 200
    assert response.json() == mock_rows_response
    mock_get_rows.assert_called_once_with(
        user_id="user-uuid-123",
        table_id=table_id,
        limit=50,  # Assuming a default limit
        offset=0,
        search_term=search_term
    )


@patch("main.row_service.create_row")
def test_create_row_success(mock_create_row, auth_headers):
    """Test successful creation of a new row."""
    table_id = 101
    row_data = {"data": {"name": "Charlie", "email": "charlie@example.com"}}
    mock_created_row = {"id": 3, "data": row_data["data"]}
    mock_create_row.return_value = mock_created_row

    response = client.post(f"/api/v1/tables/{table_id}/rows", headers=auth_headers, json=row_data)

    assert response.status_code == 201
    assert response.json() == mock_created_row
    mock_create_row.assert_called_once_with(
        user_id="user-uuid-123",
        table_id=table_id,
        row_data=row_data["data"]
    )


@patch("main.row_service.update_row")
def test_update_row_success(mock_update_row, auth_headers):
    """Test successful update of an existing row."""
    row_id = 1
    update_data = {"data": {"name": "Alice Smith", "email": "alice.smith@example.com"}}
    mock_updated_row = {"id": row_id, "data": update_data["data"]}
    mock_update_row.return_value = mock_updated_row

    response = client.put(f"/api/v1/rows/{row_id}", headers=auth_headers, json=update_data)

    assert response.status_code == 200
    assert response.json() == mock_updated_row
    mock_update_row.assert_called_once_with(
        user_id="user-uuid-123",
        row_id=row_id,
        row_data=update_data["data"]
    )


@patch("main.row_service.delete_row")
def test_delete_row_success(mock_delete_row, auth_headers):
    """Test successful deletion of a row."""
    row_id_to_delete = 2
    mock_delete_row.return_value = True

    response = client.delete(f"/api/v1/rows/{row_id_to_delete}", headers=auth_headers)

    assert response.status_code == 204
    mock_delete_row.assert_called_once_with(user_id="user-uuid-123", row_id=row_id_to_delete)


# ===================================
# Import/Export Tests
# ===================================

@patch("main.import_export_service.import_from_sql_script")
def test_import_database_from_sql(mock_import_sql, auth_headers):
    """Test creating a new database from an SQL script."""
    import_data = {
        "name": "imported_db",
        "description": "From SQL",
        "script": "CREATE TABLE test (id INT);"
    }
    mock_imported_db = {"id": 4, "name": "imported_db", "description": "From SQL"}
    mock_import_sql.return_value = mock_imported_db

    # Assuming the endpoint is /api/v1/databases/import-sql
    response = client.post("/api/v1/databases/import-sql", headers=auth_headers, json=import_data)

    assert response.status_code == 201
    assert response.json() == mock_imported_db
    mock_import_sql.assert_called_once_with(
        user_id="user-uuid-123",
        name="imported_db",
        description="From SQL",
        script="CREATE TABLE test (id INT);"
    )


@patch("main.import_export_service.export_database_to_sql")
def test_export_database_to_sql(mock_export_sql, auth_headers):
    """Test exporting a database schema and data to an SQL file."""
    db_id = 1
    sql_dump = "-- SQL Dump for database 1\nCREATE TABLE users (id SERIAL PRIMARY KEY, name TEXT);"
    mock_export_sql.return_value = sql_dump

    # Assuming the endpoint is /api/v1/databases/{db_id}/export-sql
    response = client.get(f"/api/v1/databases/{db_id}/export-sql", headers=auth_headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/sql; charset=utf-8"
    assert response.text == sql_dump
    mock_export_sql.assert_called_once_with(user_id="user-uuid-123", db_id=db_id)


@patch("main.import_export_service.preview_csv_import")
def test_preview_csv_import(mock_preview, auth_headers):
    """Test the CSV preview endpoint."""
    csv_content = "header1,header2\nvalue1,value2"
    mock_preview_data = {
        "original_headers": ["header1", "header2"],
        "sanitized_headers": ["header1", "header2"],
        "inferred_types": ["text", "text"],
    }
    mock_preview.return_value = mock_preview_data

    # Assuming endpoint is /api/v1/import-csv/preview
    response = client.post(
        "/api/v1/import-csv/preview",
        headers=auth_headers,
        json={"csv_content": csv_content}
    )

    assert response.status_code == 200
    assert response.json() == mock_preview_data
    mock_preview.assert_called_once_with(csv_content)


@patch("main.import_export_service.import_rows_from_csv")
def test_import_rows_from_csv(mock_import_rows, auth_headers):
    """Test importing rows into an existing table from a CSV."""
    table_id = 101
    import_payload = {
        "csv_content": "name,email\nDavid,david@example.com",
        "column_mapping": {"name": "name", "email": "email"}
    }
    mock_summary = {"inserted_count": 1, "failed_count": 0, "errors": []}
    mock_import_rows.return_value = mock_summary

    # Assuming endpoint is /api/v1/tables/{table_id}/import-rows-from-csv
    response = client.post(
        f"/api/v1/tables/{table_id}/import-rows-from-csv",
        headers=auth_headers,
        json=import_payload
    )

    assert response.status_code == 200
    assert response.json() == mock_summary
    mock_import_rows.assert_called_once_with(
        user_id="user-uuid-123",
        table_id=table_id,
        csv_content=import_payload["csv_content"],
        column_mapping=import_payload["column_mapping"]
    )
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, ANY

# Assume your FastAPI app is in `main.py` and is named `app`
# from main import app

# Since we don't have the real app, we'll mock it for demonstration.
# In a real scenario, you would import your actual FastAPI app.
from fastapi import FastAPI, Depends, HTTPException, status

app = FastAPI()

# Mock dependency for getting current user
async def mock_get_current_user():
    return {"id": "user-uuid-123", "email": "test@example.com"}

# This is a placeholder for your actual API endpoints.
# The tests will patch the service functions called by these endpoints.
@app.get("/api/v1/databases")
async def get_databases(current_user: dict = Depends(mock_get_current_user)):
    pass

@app.post("/api/v1/databases")
async def create_database(current_user: dict = Depends(mock_get_current_user)):
    pass

@app.delete("/api/v1/databases/{db_id}")
async def delete_database(db_id: int, current_user: dict = Depends(mock_get_current_user)):
    pass

@app.get("/api/v1/databases/by-name/{db_name}")
async def get_database_by_name(db_name: str, current_user: dict = Depends(mock_get_current_user)):
    pass

@app.post("/api/v1/databases/by-name/{db_name}/tables")
async def create_table(db_name: str, current_user: dict = Depends(mock_get_current_user)):
    pass

@app.delete("/api/v1/tables/{table_id}")
async def delete_table(table_id: int, current_user: dict = Depends(mock_get_current_user)):
    pass

@app.get("/api/v1/tables/{table_id}/rows")
async def get_rows(table_id: int, current_user: dict = Depends(mock_get_current_user)):
    pass

@app.post("/api/v1/tables/{table_id}/rows")
async def create_row(table_id: int, current_user: dict = Depends(mock_get_current_user)):
    pass

@app.put("/api/v1/rows/{row_id}")
async def update_row(row_id: int, current_user: dict = Depends(mock_get_current_user)):
    pass

@app.delete("/api/v1/rows/{row_id}")
async def delete_row(row_id: int, current_user: dict = Depends(mock_get_current_user)):
    pass

@app.post("/api/v1/users/me/delete")
async def delete_user_account(current_user: dict = Depends(mock_get_current_user)):
    pass

@app.post("/api/v1/contact")
async def contact_form():
    pass


client = TestClient(app)


@pytest.fixture
def auth_headers():
    """Provides mock authorization headers for test requests."""
    return {"Authorization": "Bearer fake-token"}


@pytest.fixture(autouse=True)
def mock_auth_dependency():
    """
    Automatically mock the `get_current_user` dependency for all tests
    to simulate an authenticated user.
    """
    app.dependency_overrides[mock_get_current_user] = lambda: {
        "id": "user-uuid-123",
        "email": "test@example.com"
    }
    yield
    app.dependency_overrides = {}


# ===================================
# User Management Tests
# ===================================

@patch("main.user_service.delete_user")
def test_delete_user_account_success(mock_delete_user, auth_headers):
    """Test successful deletion of a user account."""
    mock_delete_user.return_value = True

    response = client.post(
        "/api/v1/users/me/delete",
        headers=auth_headers,
        json={"confirmation": "delete my account"}
    )

    assert response.status_code == 200
    assert response.json() == {"message": "Account deleted successfully"}
    mock_delete_user.assert_called_once_with(user_id="user-uuid-123")


def test_delete_user_account_invalid_confirmation(auth_headers):
    """Test account deletion with incorrect confirmation text."""
    response = client.post(
        "/api/v1/users/me/delete",
        headers=auth_headers,
        json={"confirmation": "wrong text"}
    )
    # This validation is likely in the Pydantic model or endpoint logic
    # Assuming it returns a 422 Unprocessable Entity
    assert response.status_code == 422


# ===================================
# Contact Form Tests
# ===================================

@patch("main.notification_service.send_contact_email")
def test_contact_form_success(mock_send_email):
    """Test successful submission of the contact form."""
    contact_data = {
        "sender_name": "John Doe",
        "sender_email": "john.doe@example.com",
        "message": "This is a test message."
    }
    response = client.post("/api/v1/contact", json=contact_data)

    assert response.status_code == 200
    assert response.json() == {"message": "Message sent successfully"}
    mock_send_email.assert_called_once_with(
        sender_name="John Doe",
        sender_email="john.doe@example.com",
        message="This is a test message."
    )


def test_contact_form_invalid_data():
    """Test contact form submission with missing fields."""
    response = client.post("/api/v1/contact", json={"sender_name": "John Doe"})
    assert response.status_code == 422  # Unprocessable Entity


# ===================================
# Database Management Tests
# ===================================

@patch("main.db_service.get_user_databases")
def test_get_databases_success(mock_get_user_databases, auth_headers):
    """Test fetching all databases for a user."""
    mock_dbs = [
        {"id": 1, "name": "db1", "description": "First DB"},
        {"id": 2, "name": "db2", "description": "Second DB"},
    ]
    mock_get_user_databases.return_value = mock_dbs

    response = client.get("/api/v1/databases", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == mock_dbs
    mock_get_user_databases.assert_called_once_with(user_id="user-uuid-123")


@patch("main.db_service.create_database")
def test_create_database_success(mock_create_db, auth_headers):
    """Test successful creation of a new database."""
    new_db_data = {"name": "new_db", "description": "A new test database"}
    created_db = {"id": 3, **new_db_data}
    mock_create_db.return_value = created_db

    response = client.post("/api/v1/databases", headers=auth_headers, json=new_db_data)

    assert response.status_code == 201
    assert response.json() == created_db
    mock_create_db.assert_called_once_with(
        user_id="user-uuid-123",
        name="new_db",
        description="A new test database"
    )


@patch("main.db_service.create_database")
def test_create_database_duplicate_name(mock_create_db, auth_headers):
    """Test creating a database with a name that already exists."""
    mock_create_db.side_effect = HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Database with this name already exists."
    )

    response = client.post(
        "/api/v1/databases",
        headers=auth_headers,
        json={"name": "existing_db", "description": ""}
    )

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


@patch("main.db_service.delete_database")
def test_delete_database_success(mock_delete_db, auth_headers):
    """Test successful deletion of a database."""
    db_id_to_delete = 1
    mock_delete_db.return_value = True

    response = client.delete(f"/api/v1/databases/{db_id_to_delete}", headers=auth_headers)

    assert response.status_code == 204
    mock_delete_db.assert_called_once_with(user_id="user-uuid-123", db_id=db_id_to_delete)


@patch("main.db_service.delete_database")
def test_delete_database_not_found(mock_delete_db, auth_headers):
    """Test deleting a database that does not exist or user does not own."""
    mock_delete_db.side_effect = HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Database not found."
    )

    response = client.delete("/api/v1/databases/999", headers=auth_headers)

    assert response.status_code == 404
    assert response.json()["detail"] == "Database not found."


# ===================================
# Table Management Tests
# ===================================

@patch("main.db_service.get_database_by_name_with_tables")
def test_get_database_by_name(mock_get_db, auth_headers):
    """Test fetching a single database with its tables by name."""
    db_name = "my_project_db"
    mock_db_with_tables = {
        "id": 1,
        "name": db_name,
        "user_tables": [{"id": 101, "name": "users"}, {"id": 102, "name": "products"}]
    }
    mock_get_db.return_value = mock_db_with_tables

    response = client.get(f"/api/v1/databases/by-name/{db_name}", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == mock_db_with_tables
    mock_get_db.assert_called_once_with(user_id="user-uuid-123", db_name=db_name)


@patch("main.table_service.create_table")
def test_create_table_success(mock_create_table, auth_headers):
    """Test successful creation of a new table."""
    db_name = "my_project_db"
    table_data = {
        "name": "new_table",
        "columns": [
            {"name": "id", "type": "SERIAL", "is_primary_key": True},
            {"name": "name", "type": "TEXT", "is_not_null": True},
        ]
    }
    mock_created_table = {"id": 103, **table_data}
    mock_create_table.return_value = mock_created_table

    response = client.post(
        f"/api/v1/databases/by-name/{db_name}/tables",
        headers=auth_headers,
        json=table_data
    )

    assert response.status_code == 201
    assert response.json() == mock_created_table
    mock_create_table.assert_called_once_with(
        user_id="user-uuid-123",
        db_name=db_name,
        table_data=ANY  # Can assert specific fields of the Pydantic model here
    )


@patch("main.table_service.delete_table")
def test_delete_table_success(mock_delete_table, auth_headers):
    """Test successful deletion of a table."""
    table_id_to_delete = 101
    mock_delete_table.return_value = True

    response = client.delete(f"/api/v1/tables/{table_id_to_delete}", headers=auth_headers)

    assert response.status_code == 204
    mock_delete_table.assert_called_once_with(user_id="user-uuid-123", table_id=table_id_to_delete)


@patch("main.table_service.delete_table")
def test_delete_table_not_found(mock_delete_table, auth_headers):
    """Test deleting a table that doesn't exist or isn't owned by the user."""
    mock_delete_table.side_effect = HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Table not found."
    )

    response = client.delete("/api/v1/tables/9999", headers=auth_headers)

    assert response.status_code == 404


# ===================================
# Row Management Tests
# ===================================

@patch("main.row_service.get_rows_paginated")
def test_get_rows_success(mock_get_rows, auth_headers):
    """Test fetching rows from a table with pagination."""
    table_id = 101
    mock_rows_response = {
        "total": 100,
        "data": [
            {"id": 1, "data": {"name": "Alice", "email": "alice@example.com"}},
            {"id": 2, "data": {"name": "Bob", "email": "bob@example.com"}},
        ]
    }
    mock_get_rows.return_value = mock_rows_response

    response = client.get(f"/api/v1/tables/{table_id}/rows?limit=2&offset=0", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == mock_rows_response
    mock_get_rows.assert_called_once_with(
        user_id="user-uuid-123",
        table_id=table_id,
        limit=2,
        offset=0,
        search_term=None
    )


@patch("main.row_service.get_rows_paginated")
def test_get_rows_with_search(mock_get_rows, auth_headers):
    """Test fetching rows with a search term."""
    table_id = 101
    search_term = "alice"
    mock_rows_response = {
        "total": 1,
        "data": [{"id": 1, "data": {"name": "Alice", "email": "alice@example.com"}}]
    }
    mock_get_rows.return_value = mock_rows_response

    response = client.get(
        f"/api/v1/tables/{table_id}/rows?search={search_term}",
        headers=auth_headers
    )

    assert response.status_code == 200
    assert response.json() == mock_rows_response
    mock_get_rows.assert_called_once_with(
        user_id="user-uuid-123",
        table_id=table_id,
        limit=50,  # Assuming a default limit
        offset=0,
        search_term=search_term
    )


@patch("main.row_service.create_row")
def test_create_row_success(mock_create_row, auth_headers):
    """Test successful creation of a new row."""
    table_id = 101
    row_data = {"data": {"name": "Charlie", "email": "charlie@example.com"}}
    mock_created_row = {"id": 3, "data": row_data["data"]}
    mock_create_row.return_value = mock_created_row

    response = client.post(f"/api/v1/tables/{table_id}/rows", headers=auth_headers, json=row_data)

    assert response.status_code == 201
    assert response.json() == mock_created_row
    mock_create_row.assert_called_once_with(
        user_id="user-uuid-123",
        table_id=table_id,
        row_data=row_data["data"]
    )


@patch("main.row_service.update_row")
def test_update_row_success(mock_update_row, auth_headers):
    """Test successful update of an existing row."""
    row_id = 1
    update_data = {"data": {"name": "Alice Smith", "email": "alice.smith@example.com"}}
    mock_updated_row = {"id": row_id, "data": update_data["data"]}
    mock_update_row.return_value = mock_updated_row

    response = client.put(f"/api/v1/rows/{row_id}", headers=auth_headers, json=update_data)

    assert response.status_code == 200
    assert response.json() == mock_updated_row
    mock_update_row.assert_called_once_with(
        user_id="user-uuid-123",
        row_id=row_id,
        row_data=update_data["data"]
    )


@patch("main.row_service.delete_row")
def test_delete_row_success(mock_delete_row, auth_headers):
    """Test successful deletion of a row."""
    row_id_to_delete = 2
    mock_delete_row.return_value = True

    response = client.delete(f"/api/v1/rows/{row_id_to_delete}", headers=auth_headers)

    assert response.status_code == 204
    mock_delete_row.assert_called_once_with(user_id="user-uuid-123", row_id=row_id_to_delete)


# ===================================
# Import/Export Tests
# ===================================

@patch("main.import_export_service.import_from_sql_script")
def test_import_database_from_sql(mock_import_sql, auth_headers):
    """Test creating a new database from an SQL script."""
    import_data = {
        "name": "imported_db",
        "description": "From SQL",
        "script": "CREATE TABLE test (id INT);"
    }
    mock_imported_db = {"id": 4, "name": "imported_db", "description": "From SQL"}
    mock_import_sql.return_value = mock_imported_db

    # Assuming the endpoint is /api/v1/databases/import-sql
    response = client.post("/api/v1/databases/import-sql", headers=auth_headers, json=import_data)

    assert response.status_code == 201
    assert response.json() == mock_imported_db
    mock_import_sql.assert_called_once_with(
        user_id="user-uuid-123",
        name="imported_db",
        description="From SQL",
        script="CREATE TABLE test (id INT);"
    )


@patch("main.import_export_service.export_database_to_sql")
def test_export_database_to_sql(mock_export_sql, auth_headers):
    """Test exporting a database schema and data to an SQL file."""
    db_id = 1
    sql_dump = "-- SQL Dump for database 1\nCREATE TABLE users (id SERIAL PRIMARY KEY, name TEXT);"
    mock_export_sql.return_value = sql_dump

    # Assuming the endpoint is /api/v1/databases/{db_id}/export-sql
    response = client.get(f"/api/v1/databases/{db_id}/export-sql", headers=auth_headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/sql; charset=utf-8"
    assert response.text == sql_dump
    mock_export_sql.assert_called_once_with(user_id="user-uuid-123", db_id=db_id)


@patch("main.import_export_service.preview_csv_import")
def test_preview_csv_import(mock_preview, auth_headers):
    """Test the CSV preview endpoint."""
    csv_content = "header1,header2\nvalue1,value2"
    mock_preview_data = {
        "original_headers": ["header1", "header2"],
        "sanitized_headers": ["header1", "header2"],
        "inferred_types": ["text", "text"],
    }
    mock_preview.return_value = mock_preview_data

    # Assuming endpoint is /api/v1/import-csv/preview
    response = client.post(
        "/api/v1/import-csv/preview",
        headers=auth_headers,
        json={"csv_content": csv_content}
    )

    assert response.status_code == 200
    assert response.json() == mock_preview_data
    mock_preview.assert_called_once_with(csv_content)


@patch("main.import_export_service.import_rows_from_csv")
def test_import_rows_from_csv(mock_import_rows, auth_headers):
    """Test importing rows into an existing table from a CSV."""
    table_id = 101
    import_payload = {
        "csv_content": "name,email\nDavid,david@example.com",
        "column_mapping": {"name": "name", "email": "email"}
    }
    mock_summary = {"inserted_count": 1, "failed_count": 0, "errors": []}
    mock_import_rows.return_value = mock_summary

    # Assuming endpoint is /api/v1/tables/{table_id}/import-rows-from-csv
    response = client.post(
        f"/api/v1/tables/{table_id}/import-rows-from-csv",
        headers=auth_headers,
        json=import_payload
    )

    assert response.status_code == 200
    assert response.json() == mock_summary
    mock_import_rows.assert_called_once_with(
        user_id="user-uuid-123",
        table_id=table_id,
        csv_content=import_payload["csv_content"],
        column_mapping=import_payload["column_mapping"]
    )