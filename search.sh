#!/bin/bash

# Simple wrapper script for semantic search with poetry

# Check if a search query was provided
if [ $# -eq 0 ]; then
    echo "Usage: ./search.sh \"your search query\" [options]"
    echo ""
    echo "Options:"
    echo "  --top-k NUMBER      Number of files to show (default: 5)"
    echo "  --rerank-model NAME Use a specific Cohere rerank model"
    echo ""
    echo "Examples:"
    echo "  ./search.sh \"how to do a proper bandeja\" --top-k 10"
    echo "  ./search.sh \"advanced footwork\" --rerank-model rerank-v3.5"
    exit 1
fi

# Run the search through poetry
poetry run python src/semantic_search.py "$@" 