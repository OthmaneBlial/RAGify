#!/bin/bash
set -euo pipefail

# RAGify Project Linter Script
# This script runs various linting tools to keep the codebase clean

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔍 Starting RAGify Code Linting..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_tools() {
    print_status "Checking required tools..."

    local missing_tools=()

    if ! command -v autoflake &> /dev/null; then
        missing_tools+=("autoflake")
    fi

    if ! command -v ruff &> /dev/null; then
        missing_tools+=("ruff")
    fi

    if ! command -v black &> /dev/null; then
        missing_tools+=("black")
    fi

    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_warning "Missing tools: ${missing_tools[*]}"
        print_status "Installing missing tools..."
        pip install "${missing_tools[@]}"
    fi
}

# Find all Python files in the project
find_python_files() {
    find . -name "*.py" \
        -not -path "./venv/*" \
        -not -path "./.git/*" \
        -not -path "./__pycache__/*" \
        -not -path "./*.egg-info/*" \
        -not -path "./node_modules/*" \
        -not -path "./frontend/node_modules/*"
}

# Remove unused imports
remove_unused_imports() {
    print_status "Removing unused imports with autoflake..."

    local py_files
    py_files=$(find_python_files)

    if [ -z "$py_files" ]; then
        print_warning "No Python files found"
        return
    fi

    echo "$py_files" | xargs autoflake -i --remove-all-unused-imports

    print_success "Unused imports removed"
}

# Format code with Black
format_code() {
    print_status "Formatting code with Black..."

    local py_files
    py_files=$(find_python_files)

    if [ -z "$py_files" ]; then
        print_warning "No Python files found"
        return
    fi

    echo "$py_files" | xargs black --line-length 88

    print_success "Code formatted"
}

# Lint with Ruff
lint_with_ruff() {
    print_status "Linting with Ruff..."

    local py_files
    py_files=$(find_python_files)

    if [ -z "$py_files" ]; then
        print_warning "No Python files found"
        return
    fi

    echo "$py_files" | xargs ruff check --fix

    print_success "Linting completed"
}

# Check for syntax errors
check_syntax() {
    print_status "Checking Python syntax..."

    local py_files
    py_files=$(find_python_files)

    if [ -z "$py_files" ]; then
        print_warning "No Python files found"
        return
    fi

    local syntax_errors=0

    while IFS= read -r file; do
        if ! python -m py_compile "$file" 2>/dev/null; then
            print_error "Syntax error in $file"
            syntax_errors=$((syntax_errors + 1))
        fi
    done <<< "$py_files"

    if [ $syntax_errors -eq 0 ]; then
        print_success "No syntax errors found"
    else
        print_error "Found $syntax_errors files with syntax errors"
        exit 1
    fi
}

# Main linting function
run_linter() {
    check_tools
    check_syntax
    remove_unused_imports
    format_code
    lint_with_ruff

    print_success "🎉 All linting tasks completed!"
    print_status "Your codebase is now clean and properly formatted."
}

# Help function
show_help() {
    cat << EOF
RAGify Linter Script

Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    -c, --check         Run in check mode (don't modify files)
    -f, --format        Only format code with Black
    -i, --imports       Only remove unused imports
    -l, --lint          Only run Ruff linter
    -s, --syntax        Only check syntax

Examples:
    $0                  Run all linting tasks
    $0 --check          Check what would be changed without modifying
    $0 --imports        Only remove unused imports
    $0 --format         Only format code

Tools used:
    - autoflake: Remove unused imports
    - black: Code formatting
    - ruff: Fast Python linter
    - py_compile: Syntax checking

EOF
}

# Parse command line arguments
CHECK_MODE=false
ONLY_FORMAT=false
ONLY_IMPORTS=false
ONLY_LINT=false
ONLY_SYNTAX=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--check)
            CHECK_MODE=true
            shift
            ;;
        -f|--format)
            ONLY_FORMAT=true
            shift
            ;;
        -i|--imports)
            ONLY_IMPORTS=true
            shift
            ;;
        -l|--lint)
            ONLY_LINT=true
            shift
            ;;
        -s|--syntax)
            ONLY_SYNTAX=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run specific tasks or all tasks
if [ "$ONLY_SYNTAX" = true ]; then
    check_syntax
elif [ "$ONLY_IMPORTS" = true ]; then
    check_tools
    if [ "$CHECK_MODE" = true ]; then
        print_status "Checking unused imports..."
        find_python_files | xargs autoflake --check --remove-all-unused-imports
    else
        remove_unused_imports
    fi
elif [ "$ONLY_FORMAT" = true ]; then
    check_tools
    if [ "$CHECK_MODE" = true ]; then
        print_status "Checking code formatting..."
        find_python_files | xargs black --check --diff
    else
        format_code
    fi
elif [ "$ONLY_LINT" = true ]; then
    check_tools
    if [ "$CHECK_MODE" = true ]; then
        print_status "Checking linting..."
        find_python_files | xargs ruff check
    else
        lint_with_ruff
    fi
else
    if [ "$CHECK_MODE" = true ]; then
        print_status "Running in check mode - no files will be modified"
        check_tools
        check_syntax
        print_status "Checking unused imports..."
        find_python_files | xargs autoflake --check --remove-all-unused-imports || true
        print_status "Checking code formatting..."
        find_python_files | xargs black --check --diff || true
        print_status "Checking linting..."
        find_python_files | xargs ruff check || true
    else
        run_linter
    fi
fi

print_status "Linting process completed!"