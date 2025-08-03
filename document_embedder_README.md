# Document Embedder - Gemini API Semantic Search

A minimal dependency script for indexing documents and performing semantic search using Google's Gemini embedding API.

## Features

- Indexes text documents with overlapping chunks for better context preservation
- **Respects .gitignore files** - perfect for use in programming projects
- Uses Gemini's embedding API with full 3072 dimensions for accuracy
- Stores embeddings in a simple JSON file (no database required)
- Interactive REPL for querying indexed documents
- Returns top 10 most relevant snippets for any query

## Requirements

- Python 3.7+
- Only 3 external dependencies:
  - `requests` - for API calls
  - `numpy` - for cosine similarity calculations
  - `pathspec` - for proper gitignore parsing

## Installation

```bash
pip install requests numpy pathspec
```

## Setup

1. Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

2. Set your API key as an environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

## Usage

### 1. Index a folder of documents

```bash
python document_embedder.py /path/to/your/project
```

This will:
- Scan the folder recursively for text files (.txt, .md, .py, .js, .json, .html, .css, .xml, .jsx, .ts, .tsx, .yml, .yaml)
- **Respect .gitignore patterns** if present (also has sensible defaults like node_modules, venv, .git)
- Split each document into overlapping chunks (500 chars with 100 char overlap)
- Generate embeddings for each chunk using Gemini's RETRIEVAL_DOCUMENT task type
- Save everything to `document_index.json`

### 2. Search your indexed documents

```bash
python document_embedder.py
```

This opens an interactive REPL where you can:
- Type questions or keywords
- Get the top 10 most relevant snippets
- See similarity scores and source files

## Default Ignore Patterns

Even without a .gitignore file, the script ignores:
- `.git/`
- `__pycache__/`
- `*.pyc`
- `venv/`, `env/`
- `.env`
- `node_modules/`
- `*.log`
- `.DS_Store`
- `document_index.json` (the index file itself)

## Configuration

You can adjust these settings at the top of the script:

```python
EMBEDDING_DIM = 3072    # Full dimension for gemini-embedding-001
CHUNK_SIZE = 500        # Characters per chunk
OVERLAP_SIZE = 100      # Overlap between chunks
INDEX_FILE = "document_index.json"  # Where to store the index
```

## How it works

1. **Chunking**: Documents are split into overlapping chunks to preserve context across boundaries
2. **Embeddings**: Each chunk is converted to a 3072-dimensional vector using Gemini's embedding model
3. **Storage**: Embeddings are stored in a JSON file along with the original text and metadata
4. **Search**: Query embeddings use the RETRIEVAL_QUERY task type for optimal search performance
5. **Ranking**: Results are ranked by cosine similarity between query and document embeddings
6. **Gitignore**: Uses the `pathspec` library for proper gitignore pattern matching

## Example

```bash
# Index your project (respects .gitignore)
python document_embedder.py ~/projects/myapp

# Search for information
python document_embedder.py
Query> How do I configure authentication?

Top 10 results:

--- Result 1 (Score: 0.8432) ---
File: src/auth.py (Chunk 3)
Text: To configure authentication, first set up your OAuth provider...

--- Result 2 (Score: 0.7891) ---
File: docs/config.md (Chunk 7)
Text: Authentication settings can be configured in the config.yaml file...
```

## Notes

- The script uses the full 3072-dimensional embeddings (normalized) for accuracy
- Rate limiting is implemented (0.1s between API calls) to avoid hitting quotas
- Embeddings are cached in the JSON file, so you only need to index once
- The search uses the appropriate task types for optimal performance:
  - `RETRIEVAL_DOCUMENT` for indexing
  - `RETRIEVAL_QUERY` for searching
- Empty files are automatically skipped
- File paths in search results are shown as relative paths for readability

## Limitations

- JSON storage may become inefficient for very large document collections
- No incremental indexing (need to re-index entire folder if documents change)
- Basic text file support only (no PDF, DOCX, etc.)
