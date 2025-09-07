# Database Setup Guide

This guide documents the PostgreSQL setup process for the RAGify application.

## Prerequisites

- PostgreSQL installed and running
- Access to PostgreSQL superuser (postgres)

## Setup Steps

### 1. Delete Existing Database (if exists)

```bash
psql -U postgres -c "DROP DATABASE IF EXISTS ragify;"
```

### 2. Create Database User

```bash
psql -U postgres -c "CREATE USER ragify WITH PASSWORD 'RagifyStrongPass2023';"
```

### 3. Create Database

```bash
psql -U postgres -c "CREATE DATABASE ragify OWNER ragify;"
```

### 4. Enable pgvector Extension

```bash
psql -U postgres -d ragify -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 5. Update Environment Configuration

Update the `.env` file with the correct DATABASE_URL:

```env
DATABASE_URL=postgresql+asyncpg://ragify:RagifyStrongPass2023@localhost/ragify
```

### 6. Test Database Connection

```bash
PGPASSWORD='RagifyStrongPass2023' psql -U ragify -d ragify -c "SELECT version();"
```

### 7. Start the Application

```bash
./startup.sh
```

## Notes

- Use a strong password for the database user
- Ensure PostgreSQL is running before executing these commands
- The pgvector extension is required for vector operations
- The application uses asyncpg driver for PostgreSQL connections

## Troubleshooting

If you encounter connection issues:

1. Verify PostgreSQL is running: `sudo systemctl status postgresql`
2. Check user permissions: `psql -U postgres -c "\du"`
3. Test connection: Use the test command above
4. Ensure the DATABASE_URL in .env matches the created user and database

## Reset Database (if needed)

To completely reset the database setup:

```bash
# Drop database and user
psql -U postgres -c "DROP DATABASE IF EXISTS ragify;"
psql -U postgres -c "DROP USER IF EXISTS ragify;"

# Then repeat steps 2-7 above