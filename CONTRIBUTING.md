# Contributing to RAGify

Thank you for your interest in contributing to RAGify! We welcome contributions from the community to help improve and expand this Retrieval-Augmented Generation (RAG) chat application. This document provides guidelines and information to help you get started.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style and Standards](#code-style-and-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Code of Conduct](#code-of-conduct)
- [License](#license)

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- PostgreSQL 12+ with pgvector extension
- Redis 6+ (optional but recommended)
- Node.js and npm (for frontend development)
- Git

For detailed installation instructions, see the [Setup Guide](docs/setup.md).

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/your-username/RAGify.git
cd RAGify
```

3. Set up the upstream remote:

```bash
git remote add upstream https://github.com/OthmaneBlial/RAGify.git
```

## Development Setup

### Backend Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies in development mode:

```bash
pip install -e .
```

3. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Set up the database:

```bash
# Create PostgreSQL database
createdb ragify

# Enable pgvector extension
psql -d ragify -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

5. Run database migrations:

```bash
alembic upgrade head
```

### Frontend Setup

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

### Running the Application

1. Start the backend server:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

2. In another terminal, start the frontend:

```bash
cd frontend && npm run dev
```

3. Access the application:
   - Frontend: http://localhost:5173
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## Code Style and Standards

### Python Code Style

We follow PEP 8 standards with some additional conventions:

- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes using Google style
- Keep functions small and focused on single responsibilities
- Use async/await for I/O operations
- Handle errors gracefully with appropriate logging

### Code Formatting

We use the following tools for code quality:

- **Black**: For code formatting
- **isort**: For import sorting
- **flake8**: For linting

To format your code:

```bash
# Format Python code
black .

# Sort imports
isort .

# Lint code
flake8 .
```

### JavaScript/TypeScript Code Style

For frontend code:

- Use ESLint for linting
- Follow standard JavaScript/React conventions
- Use meaningful variable and function names
- Add comments for complex logic

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(auth): add JWT token validation
fix(api): resolve memory leak in chat endpoint
docs(readme): update installation instructions
```

## Testing

### Running Tests

We use pytest for testing. To run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest tests/test_api_endpoints.py

# Run tests in verbose mode
pytest -v
```

### Writing Tests

- Write comprehensive tests for new features
- Use descriptive test names that explain what they're testing
- Test both positive and negative scenarios
- Mock external dependencies when appropriate
- Aim for good test coverage (target: >80%)

### Test Structure

```
tests/
├── conftest.py              # Test configuration and fixtures
├── test_api_endpoints.py    # API endpoint tests
├── test_basic_structure.py  # Basic functionality tests
├── test_embeddings.py       # Embedding service tests
└── test_integration.py      # Integration tests
```

## Submitting Changes

### Creating a Branch

1. Create a feature branch from main:

```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit them:

```bash
git add .
git commit -m "feat: add your feature description"
```

### Pull Request Process

1. Push your branch to your fork:

```bash
git push origin feature/your-feature-name
```

2. Create a Pull Request on GitHub:
   - Use a clear, descriptive title
   - Provide a detailed description of the changes
   - Reference any related issues
   - Ensure all tests pass
   - Request review from maintainers

3. Address review feedback:
   - Make requested changes
   - Update tests if necessary
   - Rebase your branch if needed

4. Once approved, your PR will be merged by a maintainer

### Keeping Your Branch Updated

```bash
# Fetch upstream changes
git fetch upstream

# Rebase your branch on main
git rebase upstream/main

# Push the updated branch
git push origin feature/your-feature-name --force-with-lease
```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the issue
- **Steps to reproduce**: Step-by-step instructions
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: OS, Python version, browser, etc.
- **Logs**: Relevant error messages or logs
- **Screenshots**: If applicable

### Feature Requests

For feature requests, please include:

- **Description**: What feature you'd like to see
- **Use case**: Why this feature would be useful
- **Implementation ideas**: Any thoughts on how to implement it
- **Alternatives**: Other solutions you've considered

### Issue Labels

We use the following labels to categorize issues:

- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation improvements
- `question`: Questions or discussions
- `help wanted`: Good for newcomers
- `good first issue`: Suitable for first-time contributors

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. Please read and follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing to RAGify, you agree that your contributions will be licensed under the same license as the project (MIT License). See the [LICENSE](LICENSE) file for details.

## Getting Help

If you need help or have questions:

- Check the [documentation](docs/) directory
- Search existing issues on GitHub
- Ask questions in GitHub Discussions
- Join our community chat (if available)

Thank you for contributing to RAGify! Your efforts help make this project better for everyone.
