#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Please install uv first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Set PYTHONPATH to include the src directory
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Install dependencies and run in uv environment
if [ -f "pyproject.toml" ]; then
    echo "Installing dependencies with uv..."
    uv sync
    echo "Running the main script with uv..."
    uv run python main.py
else
    echo "pyproject.toml not found. Please ensure the project is properly configured for uv."
    exit 1
fi


# To run the script, make sure the script has execute permissions. To make the script executable, run:
# chmod +x run_project.sh

# Make sure uv is installed first:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# After that, you can run the script using:
# ./run_project.sh