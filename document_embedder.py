#!/usr/bin/env python3
"""
Minimal document embedding and search using Gemini API
Dependencies: requests, numpy, pathspec
"""

import json
import os
import sys
import time
from typing import List, Dict, Tuple
import requests
import numpy as np
import pathspec
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072  # Default dimension for gemini-embedding-001
CHUNK_SIZE = 3000  # Characters per chunk
OVERLAP_SIZE = 1000  # Overlap between chunks
INDEX_FILE = "document_index.json"

class DocumentEmbedder:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP_SIZE) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size // 2:  # Only break if we're past halfway
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        return [c for c in chunks if c]  # Remove empty chunks
    
    def get_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        """Get embedding for a text using Gemini API."""
        url = f"{self.base_url}/{EMBEDDING_MODEL}:embedContent"
        
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "model": f"models/{EMBEDDING_MODEL}",
            "content": {"parts": [{"text": text}]}
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            
            # Check for errors and show detailed message
            if response.status_code != 200:
                print(f"API Error {response.status_code}: {response.text}")
                return None
                
            result = response.json()
            
            # Extract embedding values
            embedding = np.array(result['embedding']['values'])
            
            # Normalize for dimensions other than 3072
            if EMBEDDING_DIM != 3072:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Error getting embedding: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None
    
    def index_folder(self, folder_path: str) -> Dict:
        """Index all text files in a folder, respecting .gitignore."""
        index = {
            "documents": [],
            "metadata": {
                "embedding_dim": EMBEDDING_DIM,
                "chunk_size": CHUNK_SIZE,
                "overlap_size": OVERLAP_SIZE,
                "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Load gitignore patterns
        gitignore_spec = None
        gitignore_path = os.path.join(folder_path, '.gitignore')
        
        # Default patterns to always ignore
        default_patterns = [
            '.git/',
            '__pycache__/',
            '*.pyc',
            'venv/',
            'env/',
            '.env',
            'node_modules/',
            '*.log',
            '.DS_Store',
            'document_index.json'  # Don't index our own index file
        ]
        
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                gitignore_patterns = f.read()
            # Combine gitignore patterns with default patterns
            all_patterns = gitignore_patterns + '\n' + '\n'.join(default_patterns)
            gitignore_spec = pathspec.PathSpec.from_lines('gitwildmatch', all_patterns.splitlines())
            print(f"Loaded .gitignore from {gitignore_path}")
        else:
            # Just use default patterns
            gitignore_spec = pathspec.PathSpec.from_lines('gitwildmatch', default_patterns)
            print("No .gitignore found, using default ignore patterns")
        
        # Get all text files
        text_extensions = ['.txt', '.md', '.py', '.js', '.json', '.html', '.css', '.xml', '.jsx', '.ts', '.tsx', '.yml', '.yaml', '.rs']
        files = []
        
        for root, dirs, filenames in os.walk(folder_path):
            # Remove ignored directories from dirs to prevent walking into them
            relative_root = os.path.relpath(root, folder_path)
            
            # Filter out directories that should be ignored
            dirs[:] = [d for d in dirs if not gitignore_spec.match_file(os.path.join(relative_root, d) + '/')]
            
            for filename in filenames:
                if any(filename.endswith(ext) for ext in text_extensions):
                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(file_path, folder_path)
                    
                    # Check if file should be ignored
                    if not gitignore_spec.match_file(relative_path):
                        files.append(file_path)
        
        print(f"Found {len(files)} files to index (after filtering)...")
        
        # Prepare all chunks first
        all_chunks = []
        for file_path in files:
            relative_path = os.path.relpath(file_path, folder_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Skip empty files
                if not content.strip():
                    continue
                
                # Chunk the document
                chunks = self.chunk_text(content)
                
                # Store chunk info for parallel processing
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "file": file_path,
                        "relative_path": relative_path,
                        "chunk_index": i,
                        "text": chunk,
                        "total_chunks": len(chunks)
                    })
                    
            except Exception as e:
                print(f"  Error reading {relative_path}: {e}")
        
        print(f"Total chunks to process: {len(all_chunks)}")
        
        # Process chunks in parallel
        def process_chunk(chunk_info):
            """Process a single chunk and return the document data"""
            try:
                embedding = self.get_embedding(chunk_info["text"], "RETRIEVAL_DOCUMENT")
                if embedding is not None:
                    return {
                        "file": chunk_info["file"],
                        "relative_path": chunk_info["relative_path"],
                        "chunk_index": chunk_info["chunk_index"],
                        "text": chunk_info["text"],
                        "embedding": embedding.tolist()
                    }
            except Exception as e:
                print(f"  Error processing chunk from {chunk_info['relative_path']}: {e}")
            return None
        
        # Use ThreadPoolExecutor for parallel processing
        processed_count = 0
        with ThreadPoolExecutor(max_workers=40) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(process_chunk, chunk): chunk 
                for chunk in all_chunks
            }
            
            # Process results as they complete
            for future in as_completed(future_to_chunk):
                chunk_info = future_to_chunk[future]
                processed_count += 1
                
                # Print progress
                if processed_count % 10 == 0 or processed_count == len(all_chunks):
                    print(f"Progress: {processed_count}/{len(all_chunks)} chunks processed")
                
                try:
                    result = future.result()
                    if result is not None:
                        index["documents"].append(result)
                except Exception as e:
                    print(f"  Error with chunk from {chunk_info['relative_path']}: {e}")
                
                # Rate limiting to avoid hitting API limits too hard
                time.sleep(0.05)  # Small delay between processing results
        
        # Save index
        print(f"\nSaving index with {len(index['documents'])} chunks...")
        with open(INDEX_FILE, 'w') as f:
            json.dump(index, f)
        
        return index
    
    def load_index(self) -> Dict:
        """Load existing index from file."""
        try:
            with open(INDEX_FILE, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def search(self, query: str, index: Dict, top_k: int = 10) -> List[Tuple[float, Dict]]:
        """Search for relevant chunks using cosine similarity."""
        # Get query embedding
        query_embedding = self.get_embedding(query, "CODE_RETRIEVAL_QUERY")
        if query_embedding is None:
            return []
        
        # Calculate similarities
        results = []
        
        for doc in index["documents"]:
            doc_embedding = np.array(doc["embedding"])
            
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding)
            results.append((similarity, doc))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[0], reverse=True)
        
        return results[:top_k]


def main():
    # Get API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        sys.exit(1)
    
    embedder = DocumentEmbedder(api_key)
    
    # Check if we need to index or search
    if len(sys.argv) > 1:
        # Index mode
        folder_path = sys.argv[1]
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            sys.exit(1)
        
        print(f"Indexing folder: {folder_path}")
        embedder.index_folder(folder_path)
        print("Indexing complete!")
        
    else:
        # Search mode - load existing index
        index = embedder.load_index()
        if not index:
            print("No index found. Please run with a folder path to create index first.")
            print(f"Usage: python {sys.argv[0]} /path/to/folder")
            sys.exit(1)
        
        print(f"Loaded index with {len(index['documents'])} chunks")
        print("Enter your questions (type 'quit' to exit):\n")
        
        # REPL loop
        while True:
            try:
                query = input("Query> ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                # Search
                print("\nSearching...")
                results = embedder.search(query, index)
                
                if not results:
                    print("No results found.")
                    continue
                
                # Display results
                print(f"\nTop {len(results)} results:\n")
                for i, (score, doc) in enumerate(results):
                    print(f"--- Result {i+1} (Score: {score:.4f}) ---")
                    # Use relative_path if available, fallback to file path
                    display_path = doc.get('relative_path', doc['file'])
                    print(f"File: {display_path} (Chunk {doc['chunk_index']})")
                    print(f"Text: {doc['text'][:200]}...")
                    print()
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
