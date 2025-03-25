# Padel Course Semantic Search

This tool allows you to search through all the Padel course video content using natural language queries. It uses semantic search with OpenAI's text-embedding-3-small model to find the most relevant video sections based on your question or topic of interest.

## Setup

1. Install the required dependencies:
   ```bash
   # Make sure liblzma is installed on your system
   # For Ubuntu/Debian: sudo apt-get install liblzma-dev
   # For macOS: brew install xz
   
   # If using pip directly
   pip install -r requirements.txt
   
   # If using Poetry (recommended)
   poetry install
   ```

2. Set up your API keys as environment variables:
   ```bash
   export OPENAI_API_KEY=your_openai_key_here
   export COHERE_API_KEY=your_cohere_key_here  # Only needed for reranking
   ```

   Alternatively, create a `.env` file in the project directory:
   ```
   OPENAI_API_KEY=your_openai_key_here
   COHERE_API_KEY=your_cohere_key_here
   ```

3. Make sure you have the video metadata files in the `resources/videos` directory.

## Usage

### Using the Wrapper Script (Recommended)

The easiest way to run searches is using the provided shell script:

```bash
./search.sh "how to do a proper bandeja"
```

With options:
```bash
./search.sh "footwork techniques" --top-k 10 --no-cache
```

### Using Poetry Directly

```bash
poetry run python src/semantic_search.py "how to do a proper bandeja"
```

### Using Python Directly

```bash
python src/semantic_search.py "how to do a proper bandeja"
```

## Command Line Options

- Specify number of files to show:
  ```bash
  ./search.sh "footwork techniques" --top-k 10
  ```

- Apply Cohere reranking (default):
  ```bash
  ./search.sh "advanced volley techniques" --rerank
  ```

- Specify a particular Cohere rerank model:
  ```bash
  ./search.sh "padel serve technique" --rerank-model rerank-v3.5
  ```

## How It Works

1. The script loads all video metadata from JSON files in the `resources/videos` directory
2. It checks for cached embeddings for each section individually
3. Only creates new embeddings for sections that have changed or are new
4. When you search:
   - Your query is converted to an embedding with the same model
   - The system calculates cosine similarity between your query and all video sections
   - Results are grouped by file and ranked by their highest scoring section
   - The top-k parameter limits the number of files returned, not the number of sections
   - With the `--rerank` option, Cohere's rerank API is used to further improve result relevance

### Section-Level Caching System

The tool implements a granular caching system to avoid redundant API calls:

- Each section's embedding is cached separately in the `cache/sections/` directory
- Metadata mapping tracks which sections exist in the current dataset
- When you run the script:
  - Only changed or new sections require new embeddings
  - Existing, unchanged sections use cached embeddings
  - All sections are tracked in a metadata file for efficient lookups
- Cache benefits:
  - Significantly reduces API costs by only processing new/changed content
  - Makes updates much faster when only some content changes
  - Each section file is named by content hash for integrity verification

## Reranking with Cohere

The system uses Cohere's reranking API to improve search results:

- Default model: `rerank-v3.5` (latest as of April 2024)
- Alternative models: `rerank-english-v2.0`, `rerank-multilingual-v2.0`
- Automatically handles different Cohere client versions (v1/v2)
- Can specify a custom model with `--rerank-model`

### V2 API Support

The tool automatically detects and uses the newer Cohere V2 API if available:
- More efficient processing
- Better handling of response formats
- Improved model stability

## Output Format

The search results include:
- Video title and section name
- Difficulty level (★ to ★★★)
- Duration of the section
- Direct link to the video with correct timestamp
- Topics covered
- Where the section is referenced in the course README

## Fallback Mode

The system includes a fallback mode that allows it to work even when Cohere is not available:
- If Cohere's API key is missing, the script will continue without reranking
- If the Cohere module cannot be imported, it will automatically disable reranking

## Example Queries

- "How do I improve my backhand technique?"
- "What equipment do I need to start playing padel?"
- "Advanced volley techniques for competitive players"
- "Footwork basics for beginners"
- "How to defend against lobs?"