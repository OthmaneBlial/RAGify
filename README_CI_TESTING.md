# Local CI Testing Guide

This guide explains how to test the GitHub Actions CI pipeline locally before pushing changes.

## Prerequisites

- Docker (for PostgreSQL and Redis containers)
- Python 3.12+
- pip

## Quick Test Script

The `test_ci_locally.sh` script simulates the entire CI pipeline locally.

### Run Full CI Simulation

```bash
./test_ci_locally.sh
```

This will:
1. Check requirements (Docker, Python, pip)
2. Start PostgreSQL with pgvector and Redis containers
3. Set up Python virtual environment
4. Install dependencies
5. Run code quality checks (Black, isort, flake8)
6. Run tests with coverage
7. Test package building
8. Clean up containers

### Expected Output

```
========================================
RAGify Local CI Simulation
========================================
[INFO] This script simulates the GitHub Actions CI pipeline locally

========================================
Checking Requirements
========================================
[SUCCESS] All required tools are available
========================================
Starting Services
========================================
[SUCCESS] Services started successfully
========================================
Setting up Python Environment
========================================
[SUCCESS] Python environment set up
========================================
Running Code Quality Checks
========================================
[SUCCESS] Code quality checks completed
========================================
Running Tests
========================================
[SUCCESS] Tests passed successfully
========================================
Testing Package Build
========================================
[SUCCESS] Package build successful
========================================
🎉 All CI Checks Passed!
========================================
```

## Manual Testing Steps

If you prefer to run individual steps manually:

### 1. Start Services

```bash
# PostgreSQL with pgvector
docker run -d --name ragify-test-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=test_ragify \
  -p 5432:5432 \
  pgvector/pgvector:pg15

# Redis
docker run -d --name ragify-test-redis \
  -p 6379:6379 \
  redis:7-alpine

# Wait for services
sleep 10

# Create vector extension
docker exec ragify-test-postgres psql -U postgres -d test_ragify -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 2. Set Up Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -e .
pip install pytest pytest-cov pytest-asyncio black isort flake8
```

### 3. Run Code Quality Checks

```bash
# Format check
black --check --diff .

# Import sorting check
isort --check-only --diff .

# Linting
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

### 4. Run Tests

```bash
# Set environment variables
export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/test_ragify"
export REDIS_URL="redis://localhost:6379"
export SECRET_KEY="test-secret-key-for-ci"
export DEBUG="true"
export OPENROUTER_API_KEY="sk-or-v1-test-key-for-ci"

# Run tests
pytest --cov=backend --cov-report=xml --cov-report=term-missing
```

### 5. Test Package Build

```bash
# Install build tools
pip install build twine setuptools wheel

# Build package
python -m build

# Check results
ls -la dist/
```

### 6. Cleanup

```bash
# Stop containers
docker stop ragify-test-postgres ragify-test-redis

# Remove containers
docker rm ragify-test-postgres ragify-test-redis

# Remove virtual environment and build artifacts
rm -rf venv dist/
```

## Troubleshooting

### Common Issues

1. **Port already in use**: Stop existing containers or change ports
   ```bash
   docker ps
   docker stop <container_id>
   ```

2. **Permission denied**: Make script executable
   ```bash
   chmod +x test_ci_locally.sh
   ```

3. **Python version issues**: Ensure Python 3.12+ is used
   ```bash
   python3 --version
   ```

4. **Docker not available**: Install Docker or use alternative testing method

### Environment Variables

The test script uses these environment variables (matching CI):

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: Django-style secret key
- `DEBUG`: Debug mode flag
- `OPENROUTER_API_KEY`: API key for testing (dummy value)

## Integration with Development Workflow

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
echo "Running local CI checks..."
./test_ci_locally.sh
if [ $? -ne 0 ]; then
    echo "CI checks failed. Please fix issues before committing."
    exit 1
fi
```

### VS Code Tasks

Add to `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Run Local CI",
            "type": "shell",
            "command": "./test_ci_locally.sh",
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        }
    ]
}
```

## What the Script Tests

The local CI simulation verifies:

1. **Dependencies**: All required packages can be installed
2. **Database**: PostgreSQL with pgvector works correctly
3. **Caching**: Redis connection is functional
4. **Code Quality**: Black, isort, and flake8 checks pass
5. **Tests**: All tests run successfully with coverage
6. **Packaging**: Package can be built for PyPI release

This ensures that the GitHub Actions CI pipeline will pass when you push your changes.