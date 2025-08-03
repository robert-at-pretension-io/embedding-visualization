#!/usr/bin/env python3
"""
Interactive visualization server for document embeddings
"""

import json
import os
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from flask_caching import Cache
from json import JSONEncoder
import time
from typing import Dict, Any, Optional, List, Tuple
import threading
import uuid
import requests
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pathspec

# Import our visualization components
from visualize_embeddings import EmbeddingVisualizer
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import plotly.graph_objects as go

class NumpyJSONEncoder(JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Create custom JSON provider for Flask 2.2+
from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app = Flask(__name__)
# Set the custom JSON provider for Flask 2.2+
app.json = NumpyJSONProvider(app)
CORS(app)

# Configure caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Global storage for computation status
computation_status = {}

# Gemini API configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Configuration for embeddings
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072  # Default dimension for gemini-embedding-001
CHUNK_SIZE = 3000  # Characters per chunk
OVERLAP_SIZE = 1000  # Overlap between chunks

class DocumentEmbedder:
    """Handles document embedding using Gemini API"""
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
            
            if response.status_code != 200:
                print(f"API Error {response.status_code}: {response.text}")
                return None
                
            result = response.json()
            embedding = np.array(result['embedding']['values'])
            
            # Normalize for dimensions other than 3072
            if EMBEDDING_DIM != 3072:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def search(self, query: str, index_data: Dict, top_k: int = 10) -> List[Tuple[float, Dict]]:
        """Search for relevant chunks using cosine similarity."""
        # Get query embedding
        query_embedding = self.get_embedding(query, "CODE_RETRIEVAL_QUERY")
        if query_embedding is None:
            return []
        
        # Calculate similarities
        results = []
        
        for doc in index_data["documents"]:
            doc_embedding = np.array(doc["embedding"])
            
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding)
            results.append((similarity, doc))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[0], reverse=True)
        
        return results[:top_k]
    
    def index_folder(self, folder_path: str, progress_callback=None) -> Dict:
        """Index all text files in a folder, respecting .gitignore."""
        index = {
            "documents": [],
            "metadata": {
                "embedding_dim": EMBEDDING_DIM,
                "chunk_size": CHUNK_SIZE,
                "overlap_size": OVERLAP_SIZE,
                "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "folder_path": folder_path
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
            'document_index.json',
            'indices/'  # Don't index our indices directory
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
        
        # Update metadata
        index["metadata"]["total_files"] = len(files)
        
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
        index["metadata"]["total_chunks"] = len(all_chunks)
        
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
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(process_chunk, chunk): chunk 
                for chunk in all_chunks
            }
            
            # Process results as they complete
            for future in as_completed(future_to_chunk):
                chunk_info = future_to_chunk[future]
                processed_count += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(processed_count, len(all_chunks))
                
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
        
        return index

# Configuration file paths
CONFIG_FILE = "visualization_configs.json"
CLUSTER_STATES_FILE = "cluster_states.json"
INDICES_DIR = "indices"
INDEX_REGISTRY_FILE = "index_registry.json"

def load_configurations():
    """Load saved configurations from file"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading configurations: {e}")
    return {
        "configurations": {},
        "default": None
    }

def sanitize_for_json(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    return obj

def load_index_registry():
    """Load the index registry that tracks all available indices"""
    if os.path.exists(INDEX_REGISTRY_FILE):
        try:
            with open(INDEX_REGISTRY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading index registry: {e}")
    return {
        "indices": {},
        "default_index": None,
        "current_index": None
    }

def save_index_registry(registry):
    """Save the index registry"""
    try:
        sanitized_registry = sanitize_for_json(registry)
        with open(INDEX_REGISTRY_FILE, 'w') as f:
            json.dump(sanitized_registry, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving index registry: {e}")
        return False

def ensure_indices_directory():
    """Ensure the indices directory exists"""
    if not os.path.exists(INDICES_DIR):
        os.makedirs(INDICES_DIR)
        print(f"Created indices directory: {INDICES_DIR}")

def save_configurations(configs):
    """Save configurations to file"""
    try:
        # Sanitize numpy types before saving
        sanitized_configs = sanitize_for_json(configs)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(sanitized_configs, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving configurations: {e}")
        return False

def load_cluster_states():
    """Load saved cluster states from file"""
    if os.path.exists(CLUSTER_STATES_FILE):
        try:
            with open(CLUSTER_STATES_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cluster states: {e}")
    return {'states': {}, 'last_used': None}

def save_cluster_states(states):
    """Save cluster states to file"""
    try:
        # Sanitize numpy types before saving
        sanitized_states = sanitize_for_json(states)
        with open(CLUSTER_STATES_FILE, 'w') as f:
            json.dump(sanitized_states, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving cluster states: {e}")
        return False

def validate_cluster_state(state, max_doc_index):
    """Validate a cluster state against current document index
    
    Args:
        state: The cluster state to validate
        max_doc_index: Maximum valid document index
        
    Returns:
        Tuple of (is_valid, cleaned_state, validation_errors)
    """
    validation_errors = []
    cleaned_state = state.copy()
    
    # Validate hierarchical summaries
    if 'hierarchical_summaries' in cleaned_state:
        for path, summary_data in list(cleaned_state['hierarchical_summaries'].items()):
            if 'summaries' in summary_data:
                for cluster_id, cluster_info in list(summary_data['summaries'].items()):
                    # Check if cluster references valid documents
                    if 'original_indices' in cluster_info:
                        valid_indices = [idx for idx in cluster_info['original_indices'] 
                                       if 0 <= idx <= max_doc_index]
                        if len(valid_indices) != len(cluster_info['original_indices']):
                            validation_errors.append(
                                f"Cluster {cluster_id} in path {path} references invalid document indices"
                            )
                        cluster_info['original_indices'] = valid_indices
                        
                    # Remove cluster if no valid documents
                    if 'doc_count' in cluster_info and cluster_info['doc_count'] == 0:
                        del summary_data['summaries'][cluster_id]
                        validation_errors.append(f"Removed empty cluster {cluster_id} from path {path}")
    
    # Validate navigation tree
    if 'navigation_tree' in cleaned_state:
        for path, nav_data in list(cleaned_state['navigation_tree'].items()):
            if 'cluster_info' in nav_data and 'original_indices' in nav_data['cluster_info']:
                valid_indices = [idx for idx in nav_data['cluster_info']['original_indices'] 
                               if 0 <= idx <= max_doc_index]
                if len(valid_indices) == 0:
                    del cleaned_state['navigation_tree'][path]
                    validation_errors.append(f"Removed invalid navigation tree entry for path {path}")
                else:
                    nav_data['cluster_info']['original_indices'] = valid_indices
                    nav_data['cluster_info']['doc_count'] = len(valid_indices)
    
    # Validate cluster history
    if 'cluster_history' in cleaned_state:
        valid_history = []
        for hist_item in cleaned_state['cluster_history']:
            if 'data' in hist_item and hist_item['data']:
                # Basic validation - just ensure it has required fields
                if 'plot' in hist_item['data']:
                    valid_history.append(hist_item)
                else:
                    validation_errors.append(f"Removed invalid history item for path {hist_item.get('path', 'unknown')}")
        cleaned_state['cluster_history'] = valid_history
    
    is_valid = len(validation_errors) == 0
    return is_valid, cleaned_state, validation_errors

class InteractiveVisualizer(EmbeddingVisualizer):
    """Extended visualizer with support for multiple algorithms"""
    
    def compute_reduction(self, algorithm: str, params: Dict[str, Any], 
                         cache_key: Optional[str] = None) -> np.ndarray:
        """Compute dimensionality reduction with specified algorithm and parameters"""
        
        # Check cache first
        if cache_key:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Standardize features
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings)
        
        # Apply selected algorithm
        if algorithm == 'tsne':
            reducer = TSNE(
                n_components=params.get('n_components', 2),
                perplexity=params.get('perplexity', 30),
                learning_rate=params.get('learning_rate', 200),
                max_iter=params.get('n_iter', 1000),
                early_exaggeration=params.get('early_exaggeration', 12),
                random_state=42,
                verbose=1
            )
        elif algorithm == 'umap':
            reducer = umap.UMAP(
                n_components=params.get('n_components', 2),
                n_neighbors=params.get('n_neighbors', 15),
                min_dist=params.get('min_dist', 0.1),
                metric=params.get('metric', 'euclidean'),
                random_state=42
            )
        elif algorithm == 'pca':
            reducer = PCA(
                n_components=params.get('n_components', 2),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Compute reduction
        result = reducer.fit_transform(embeddings_scaled)
        
        # Cache result
        if cache_key:
            cache.set(cache_key, result, timeout=3600)  # Cache for 1 hour
        
        return result
    
    def apply_clustering(self, coordinates: np.ndarray, algorithm: str, params: Dict[str, Any]) -> np.ndarray:
        """Apply clustering algorithm to coordinates"""
        
        if algorithm == 'kmeans':
            n_clusters = params.get('n_clusters', 5)
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(coordinates)
            
        elif algorithm == 'dbscan':
            eps = params.get('eps', 0.5)
            min_samples = params.get('min_samples', 5)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(coordinates)
            
        elif algorithm == 'hierarchical':
            n_clusters = params.get('n_clusters', 5)
            linkage = params.get('linkage', 'ward')
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            labels = clusterer.fit_predict(coordinates)
            
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
            
        return labels
    
    def apply_recursive_clustering(self, 
                                 indices: Optional[List[int]] = None,
                                 parent_id: str = "root",
                                 level: int = 0,
                                 max_depth: int = 3,
                                 algorithm: str = 'kmeans',
                                 dim_reduction_algorithm: str = 'umap',
                                 cluster_params: Dict[str, Any] = None,
                                 dim_reduction_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply clustering recursively to create a hierarchy"""
        
        if cluster_params is None:
            cluster_params = {}
        if dim_reduction_params is None:
            dim_reduction_params = {}
            
        # If no indices provided, use all documents
        if indices is None:
            indices = list(range(len(self.documents)))
        
        # Base case: max depth reached or too few documents
        if level >= max_depth or len(indices) < max(10, cluster_params.get('min_samples', 5) * 2):
            return {
                "cluster_id": parent_id,
                "level": level,
                "doc_indices": indices,
                "doc_count": len(indices),
                "children": []
            }
        
        # Get embeddings for this subset
        subset_embeddings = self.embeddings[indices]
        
        # Apply dimensionality reduction
        if len(indices) > 3:  # Need at least 3 points for most algorithms
            try:
                # We need to apply reduction only to the subset
                # Temporarily store original embeddings
                original_embeddings = self.embeddings
                original_documents = self.documents
                
                # Set subset as current data
                self.embeddings = subset_embeddings
                self.documents = [self.documents[i] for i in indices]
                
                # Create a new params dict for this level
                reduction_params = {
                    'algorithm': dim_reduction_algorithm,
                    'n_components': 2,  # Always use 2D for sub-clustering
                    **dim_reduction_params
                }
                
                # Compute reduction for this subset
                subset_coordinates = self.compute_reduction(
                    dim_reduction_algorithm, 
                    reduction_params,
                    cache_key=f"{parent_id}_{level}_{dim_reduction_algorithm}"
                )
                
                # Restore original data
                self.embeddings = original_embeddings
                self.documents = original_documents
                
            except Exception as e:
                print(f"Error in dimensionality reduction at level {level}: {e}")
                # Fall back to using raw embeddings
                subset_coordinates = subset_embeddings[:, :2]  # Just use first 2 dimensions
        else:
            subset_coordinates = subset_embeddings[:, :2]
        
        # Apply clustering
        if len(indices) > cluster_params.get('n_clusters', 5):
            labels = self.apply_clustering(subset_coordinates, algorithm, cluster_params)
        else:
            # Too few points to cluster further
            return {
                "cluster_id": parent_id,
                "level": level,
                "doc_indices": indices,
                "doc_count": len(indices),
                "children": []
            }
        
        # Build hierarchy node
        node = {
            "cluster_id": parent_id,
            "level": level,
            "doc_indices": indices,
            "doc_count": len(indices),
            "children": [],
            "cluster_params": cluster_params,
            "coordinates": subset_coordinates.tolist()  # Store for visualization
        }
        
        # Group indices by cluster label
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == -1:  # Skip outliers in DBSCAN
                continue
                
            # Get indices in this cluster
            cluster_mask = labels == label
            cluster_indices = [indices[i] for i, mask in enumerate(cluster_mask) if mask]
            
            # Create child cluster ID
            child_id = f"{parent_id}-{label}" if parent_id != "root" else str(label)
            
            # Recursively cluster this subset
            child_node = self.apply_recursive_clustering(
                indices=cluster_indices,
                parent_id=child_id,
                level=level + 1,
                max_depth=max_depth,
                algorithm=algorithm,
                dim_reduction_algorithm=dim_reduction_algorithm,
                cluster_params=cluster_params,
                dim_reduction_params=dim_reduction_params
            )
            
            node["children"].append(child_node)
        
        return node
    
    def summarize_clusters(self, coordinates: np.ndarray, labels: np.ndarray, subset_indices: np.ndarray = None) -> Dict[int, Dict[str, Any]]:
        """Generate summaries for each cluster using Gemini API with parallel requests
        
        Args:
            coordinates: The reduced coordinates for visualization
            labels: Cluster labels for each document
            subset_indices: Optional array of indices to limit summarization to a subset of documents
        """
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        print(f"Using Gemini API with key: {GEMINI_API_KEY[:10]}...")
        print(f"API URL: {GEMINI_API_URL}")
        
        summaries = {}
        unique_labels = np.unique(labels)
        
        def summarize_single_cluster(label: int) -> Tuple[int, Dict[str, Any]]:
            """Helper function to summarize a single cluster"""
            if label == -1:  # Skip outliers in DBSCAN
                return None, None
                
            # Get documents in this cluster
            indices = np.where(labels == label)[0]
            
            # If subset_indices is provided, filter to only include documents in the subset
            if subset_indices is not None:
                # Map indices to the subset
                indices = np.array([i for i in indices if i < len(subset_indices)])
                if len(indices) == 0:
                    return None, None
                # Get the actual document indices from the subset
                actual_indices = subset_indices[indices]
            else:
                actual_indices = indices
                
            cluster_docs = [self.documents[i] for i in actual_indices]
            
            # Prepare text for summarization
            doc_texts = []
            file_refs = {}
            
            for doc in cluster_docs[:20]:  # Limit to 20 docs to avoid token limits
                file_path = doc['relative_path']
                chunk_idx = doc['chunk_index']
                
                if file_path not in file_refs:
                    file_refs[file_path] = []
                file_refs[file_path].append(chunk_idx)
                
                # Include file info and truncated text
                doc_texts.append(
                    f"File: {file_path}, Chunk {chunk_idx}:\n{doc['text'][:300]}..."
                )
            
            # Create prompt
            cluster_text = "\n\n".join(doc_texts)
            prompt = f"""Analyze these {len(cluster_docs)} document snippets from a cluster and identify their main theme or commonality. 
Provide a summary in 5 sentences or less that explains what connects these documents.

Documents:
{cluster_text}"""
            
            # Call Gemini API
            try:
                headers = {
                    'x-goog-api-key': GEMINI_API_KEY,
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 100000
                    }
                }
                
                response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
                print(f"Cluster {label} - Gemini API response status: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"Cluster {label} - Gemini API error: {response.text}")
                    response.raise_for_status()
                
                result = response.json()
                
                # Check if we have valid candidates
                if 'candidates' not in result or not result['candidates']:
                    raise ValueError("No candidates in Gemini response")
                
                summary_text = result['candidates'][0]['content']['parts'][0]['text']
                
                # Generate a name for the cluster based on the summary
                cluster_name = f"Cluster {label}"  # Default fallback
                try:
                    name_prompt = f"""Generate a 3-word name for this cluster. Output ONLY the name, nothing else.

Example: "Machine Learning Models"

Summary: {summary_text}

Name:"""
                    
                    name_payload = {
                        "contents": [{
                            "parts": [{
                                "text": name_prompt
                            }]
                        }],
                        "generationConfig": {
                            "temperature": 0.1,  # Lower temperature for consistent naming
                            "maxOutputTokens": 100000  # Increased to allow for internal processing
                        }
                    }
                    
                    name_response = requests.post(GEMINI_API_URL, headers=headers, json=name_payload)
                    
                    if name_response.status_code == 200:
                        name_result = name_response.json()
                        print(f"Cluster {label} name API response: {name_result}")  # Debug logging
                        
                        if 'candidates' in name_result and name_result['candidates']:
                            generated_name = name_result['candidates'][0]['content']['parts'][0]['text'].strip()
                            
                            # Validate the name is not empty
                            if generated_name and not generated_name.isspace():
                                # Clean up the name - remove quotes if present
                                generated_name = generated_name.strip('"\'')
                                
                                # Ensure it's not too long (fallback to first 3 words if needed)
                                words = generated_name.split()
                                if len(words) > 3:
                                    cluster_name = ' '.join(words[:3])
                                else:
                                    cluster_name = generated_name
                                    
                                print(f"Generated name for cluster {label}: '{cluster_name}'")
                            else:
                                print(f"Empty name generated for cluster {label}, using fallback")
                    else:
                        print(f"Failed to generate name for cluster {label}: {name_response.status_code}")
                        
                except Exception as e:
                    print(f"Error generating name for cluster {label}: {str(e)}")
                    # Keep default fallback name
                
                # Format file references
                file_list = []
                for file, chunks in file_refs.items():
                    if len(chunks) == 1:
                        file_list.append(f"{file} (chunk {chunks[0]})")
                    else:
                        chunk_range = f"{min(chunks)}-{max(chunks)}" if max(chunks) - min(chunks) > 1 else f"{chunks[0]}, {chunks[1]}"
                        file_list.append(f"{file} (chunks {chunk_range})")
                
                return int(label), {
                    'summary': summary_text.strip(),
                    'name': cluster_name,
                    'doc_count': len(cluster_docs),
                    'files': file_list[:10],  # Limit file list
                    'cluster_id': int(label)
                }
                
            except Exception as e:
                print(f"Error generating summary for cluster {label}: {str(e)}")
                
                return int(label), {
                    'summary': f"Error generating summary: {str(e)}",
                    'name': f"Cluster {label}",  # Fallback name
                    'doc_count': len(cluster_docs),
                    'files': list(file_refs.keys())[:10],
                    'cluster_id': int(label)
                }
        
        # Process clusters in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_label = {
                executor.submit(summarize_single_cluster, label): label 
                for label in unique_labels if label != -1
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_label):
                label, summary_data = future.result()
                if label is not None and summary_data is not None:
                    summaries[label] = summary_data
        
        print(f"Generated {len(summaries)} summaries")
        return summaries
    
    def create_plot_data(self, coordinates: np.ndarray, color_by: str = 'file', 
                        cluster_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create plot data in JSON format for frontend"""
        import plotly.express as px
        print(f"[CREATE_PLOT] Starting with color_by: {color_by}, coordinates shape: {coordinates.shape}")
        traces = []
        
        if color_by == 'file':
            colors = px.colors.qualitative.Plotly
            
            # Group by file
            file_groups = {}
            for i, doc in enumerate(self.documents):
                file = doc['relative_path']
                if file not in file_groups:
                    file_groups[file] = []
                file_groups[file].append(i)
            
            # Create trace for each file
            for idx, (file, indices) in enumerate(file_groups.items()):
                color = colors[idx % len(colors)]
                # Convert indices to regular Python ints
                indices = [int(i) for i in indices]
                
                trace = {
                    'x': coordinates[indices, 0].tolist(),
                    'y': coordinates[indices, 1].tolist(),
                    'mode': 'markers',
                    'type': 'scatter',
                    'name': file if len(file) < 30 else file[:27] + '...',
                    'marker': {
                        'size': 8,
                        'color': color,
                        'opacity': 0.7
                    },
                    'text': [f"File: {self.documents[i]['relative_path']}<br>"
                            f"Chunk: {int(self.documents[i]['chunk_index'])}<br>"
                            f"Preview: {self.documents[i]['text'][:200]}..."
                            for i in indices],
                    'hovertemplate': '%{text}<extra></extra>'
                }
                
                if coordinates.shape[1] > 2:  # 3D
                    trace['z'] = coordinates[indices, 2].tolist()
                    trace['type'] = 'scatter3d'
                
                traces.append(trace)
                
        elif color_by in ['kmeans', 'dbscan', 'hierarchical']:
            # Apply clustering
            if cluster_params is None:
                cluster_params = {}
            
            labels = self.apply_clustering(coordinates, color_by, cluster_params)
            
            # Get unique labels and assign colors
            unique_labels = np.unique(labels)
            colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set3
            
            # Create trace for each cluster
            for label in unique_labels:
                indices = np.where(labels == label)[0]
                # Convert to regular Python list of ints
                indices = [int(i) for i in indices]
                
                # Convert label to regular int
                label = int(label)
                
                # Handle outliers in DBSCAN (label = -1)
                if label == -1:
                    color = 'gray'
                    name = 'Outliers'
                    marker_symbol = 'x'
                else:
                    color = colors[label % len(colors)]
                    name = f'Cluster {label}'
                    marker_symbol = 'circle'
                
                trace = {
                    'x': coordinates[indices, 0].tolist(),
                    'y': coordinates[indices, 1].tolist(),
                    'mode': 'markers',
                    'type': 'scatter',
                    'name': name,
                    'marker': {
                        'size': 8,
                        'color': color,
                        'opacity': 0.7,
                        'symbol': marker_symbol
                    },
                    'text': [f"Cluster: {name}<br>"
                            f"File: {self.documents[i]['relative_path']}<br>"
                            f"Chunk: {int(self.documents[i]['chunk_index'])}<br>"
                            f"Preview: {self.documents[i]['text'][:200]}..."
                            for i in indices],
                    'customdata': [[label] for _ in indices],  # Add cluster ID for click handling
                    'hovertemplate': '%{text}<extra></extra>'
                }
                
                if coordinates.shape[1] > 2:  # 3D
                    trace['z'] = coordinates[indices, 2].tolist()
                    trace['type'] = 'scatter3d'
                
                traces.append(trace)
        
        return {
            'data': traces,
            'layout': {
                'title': f'Document Embeddings ({len(self.documents)} chunks)',
                'xaxis': {'title': 'Component 1'},
                'yaxis': {'title': 'Component 2'},
                'hovermode': 'closest',
                'width': 900,
                'height': 700,
                'template': 'plotly_white'
            }
        }

# Global visualizer instance and index management
visualizer = None
document_index_hash = None
current_index_name = None
index_registry = None
embedder = None
indexing_progress = {}

def compute_document_index_hash():
    """Compute a hash of the document index for validation"""
    try:
        with open("document_index.json", 'r') as f:
            index_data = json.load(f)
        
        # Create a stable hash of document count and total size
        doc_count = len(index_data.get('documents', []))
        metadata = index_data.get('metadata', {})
        
        # Create a string representation of key metrics
        hash_input = f"{doc_count}:{metadata.get('total_chunks', 0)}:{metadata.get('total_files', 0)}"
        
        # Add first few document IDs for additional validation
        docs = index_data.get('documents', [])
        for i in range(min(10, len(docs))):
            hash_input += f":{docs[i].get('relative_path', '')}:{int(docs[i].get('chunk_index', 0))}"
        
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    except Exception as e:
        print(f"Error computing document index hash: {e}")
        return None

def initialize_visualizer(index_file=None):
    """Initialize the visualizer with document data from specified index"""
    global visualizer, document_index_hash, current_index_name, index_registry, embedder
    
    # Initialize embedder if not already done
    if embedder is None and GEMINI_API_KEY:
        embedder = DocumentEmbedder(GEMINI_API_KEY)
    
    # Ensure indices directory exists
    ensure_indices_directory()
    
    # Load index registry
    index_registry = load_index_registry()
    
    # Determine which index to load
    if index_file is None:
        # Check for default index
        if index_registry.get("default_index"):
            index_name = index_registry["default_index"]
            if index_name in index_registry.get("indices", {}):
                index_file = index_registry["indices"][index_name]["file_path"]
                current_index_name = index_name
        
        # Fall back to document_index.json if no default
        if index_file is None:
            index_file = "document_index.json"
            current_index_name = "default"
    
    # Initialize visualizer with the index file
    visualizer = InteractiveVisualizer(index_file)
    if not visualizer.load_index():
        # If the specified index doesn't exist, try the default document_index.json
        if index_file != "document_index.json" and os.path.exists("document_index.json"):
            print(f"Failed to load {index_file}, falling back to document_index.json")
            visualizer = InteractiveVisualizer("document_index.json")
            if not visualizer.load_index():
                raise Exception("Failed to load any document index")
            current_index_name = "default"
        else:
            raise Exception(f"Failed to load document index from {index_file}")
    
    # Compute document index hash for validation
    document_index_hash = compute_document_index_hash()
    
    # Update registry with current index
    index_registry["current_index"] = current_index_name
    save_index_registry(index_registry)

@app.route('/')
def index():
    """Serve the interactive visualization interface"""
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive Document Embeddings Visualization</title>
        <meta charset="utf-8">
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .controls {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 8px;
            }
            .control-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #333;
            }
            select, input[type="range"], input[type="number"] {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }
            input[type="range"] {
                padding: 0;
            }
            .range-value {
                display: inline-block;
                margin-left: 10px;
                font-weight: normal;
                color: #666;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
            }
            button:hover {
                background-color: #45a049;
            }
            button:disabled {
                background-color: #cccccc;
                cursor: not-allowed;
            }
            #plot {
                margin-top: 20px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }
            .status {
                margin-top: 10px;
                padding: 10px;
                border-radius: 4px;
                display: none;
            }
            .status.loading {
                background-color: #fff3cd;
                color: #856404;
                display: block;
            }
            .status.error {
                background-color: #f8d7da;
                color: #721c24;
                display: block;
            }
            .status.success {
                background-color: #d4edda;
                color: #155724;
                display: block;
            }
            .algorithm-params {
                display: none;
            }
            .algorithm-params.active {
                display: block;
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
            }
            .info {
                color: #666;
                margin-bottom: 20px;
            }
            .cluster-summary {
                margin-bottom: 20px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 8px;
                border-left: 4px solid #4CAF50;
            }
            .cluster-summary h3 {
                margin-top: 0;
                color: #333;
            }
            .cluster-summary .summary-text {
                margin: 10px 0;
                line-height: 1.6;
            }
            .cluster-summary .file-list {
                font-size: 0.9em;
                color: #666;
                margin-top: 10px;
            }
            .cluster-summary .file-list strong {
                color: #333;
            }
            .cluster-checkbox {
                float: right;
                transform: scale(1.5);
                cursor: pointer;
            }
            .delete-clusters-btn {
                background-color: #dc3545;
                margin-left: 10px;
            }
            .delete-clusters-btn:hover {
                background-color: #c82333;
            }
            .delete-warning {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                color: #856404;
                padding: 10px;
                border-radius: 4px;
                margin-bottom: 15px;
                display: none;
            }
            .minimize-btn {
                position: absolute;
                top: 10px;
                right: 40px;
                background: none;
                border: none;
                font-size: 20px;
                cursor: pointer;
                color: #666;
            }
            .minimize-btn:hover {
                color: #333;
            }
            #show-summaries-btn {
                background-color: #17a2b8;
                margin-left: 10px;
                display: none;
            }
            #show-summaries-btn:hover {
                background-color: #138496;
            }
            .config-controls {
                margin-top: 20px;
                padding: 15px;
                background-color: #f0f0f0;
                border-radius: 8px;
                display: flex;
                gap: 10px;
                align-items: center;
                flex-wrap: wrap;
            }
            .config-select {
                min-width: 200px;
            }
            .save-config-btn {
                background-color: #28a745;
            }
            .save-config-btn:hover {
                background-color: #218838;
            }
            .delete-config-btn {
                background-color: #dc3545;
                padding: 8px 16px;
            }
            .delete-config-btn:hover {
                background-color: #c82333;
            }
            .default-indicator {
                color: #28a745;
                font-weight: bold;
                margin-left: 5px;
            }
            .cluster-item {
                padding: 10px;
                margin: 5px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .cluster-item:hover {
                background-color: #f5f5f5;
            }
            .cluster-item input[type="checkbox"] {
                margin-right: 10px;
                transform: scale(1.2);
            }
            .cluster-stats {
                font-size: 0.9em;
                color: #666;
            }
            .breadcrumb {
                padding: 10px 15px;
                background-color: #f8f9fa;
                border-radius: 4px;
                margin-bottom: 15px;
                display: none;
            }
            .breadcrumb.active {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .breadcrumb-item {
                cursor: pointer;
                color: #007bff;
                text-decoration: none;
            }
            .breadcrumb-item:hover {
                text-decoration: underline;
            }
            .breadcrumb-separator {
                color: #6c757d;
            }
            .drill-down-info {
                background-color: #d1ecf1;
                border: 1px solid #bee5eb;
                color: #0c5460;
                padding: 10px;
                border-radius: 4px;
                margin-bottom: 10px;
                display: none;
            }
            .drill-down-info.active {
                display: block;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Interactive Document Embeddings Visualization</h1>
            <p class="info">Adjust parameters below to explore different clustering configurations</p>
            
            <div class="controls">
                <div>
                    <div class="control-group">
                        <label for="algorithm">Algorithm:</label>
                        <select id="algorithm" onchange="updateAlgorithmParams()">
                            <option value="tsne">t-SNE</option>
                            <option value="umap">UMAP</option>
                            <option value="pca">PCA</option>
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <label for="dimensions">Dimensions:</label>
                        <select id="dimensions">
                            <option value="2">2D</option>
                            <option value="3">3D</option>
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <label for="color_by">Color by:</label>
                        <select id="color_by" onchange="updateColorByParams()">
                            <option value="file">File</option>
                            <option value="kmeans">K-Means Clustering</option>
                            <option value="dbscan">DBSCAN Clustering</option>
                            <option value="hierarchical">Hierarchical Clustering</option>
                        </select>
                    </div>
                </div>
                
                <div>
                    <!-- t-SNE parameters -->
                    <div id="tsne-params" class="algorithm-params active">
                        <div class="control-group">
                            <label for="tsne-perplexity">
                                Perplexity: <span class="range-value" id="tsne-perplexity-value">30</span>
                            </label>
                            <input type="range" id="tsne-perplexity" min="5" max="50" value="30" step="1"
                                   oninput="updateRangeValue('tsne-perplexity')">
                        </div>
                        
                        <div class="control-group">
                            <label for="tsne-learning-rate">
                                Learning Rate: <span class="range-value" id="tsne-learning-rate-value">200</span>
                            </label>
                            <input type="range" id="tsne-learning-rate" min="10" max="1000" value="200" step="10"
                                   oninput="updateRangeValue('tsne-learning-rate')">
                        </div>
                        
                        <div class="control-group">
                            <label for="tsne-iterations">
                                Iterations: <span class="range-value" id="tsne-iterations-value">1000</span>
                            </label>
                            <input type="range" id="tsne-iterations" min="250" max="5000" value="1000" step="250"
                                   oninput="updateRangeValue('tsne-iterations')">
                        </div>
                        
                        <div class="control-group">
                            <label for="tsne-early-exaggeration">
                                Early Exaggeration: <span class="range-value" id="tsne-early-exaggeration-value">12</span>
                            </label>
                            <input type="range" id="tsne-early-exaggeration" min="4" max="50" value="12" step="2"
                                   oninput="updateRangeValue('tsne-early-exaggeration')">
                        </div>
                    </div>
                    
                    <!-- UMAP parameters -->
                    <div id="umap-params" class="algorithm-params">
                        <div class="control-group">
                            <label for="umap-neighbors">
                                N Neighbors: <span class="range-value" id="umap-neighbors-value">15</span>
                            </label>
                            <input type="range" id="umap-neighbors" min="5" max="50" value="15" step="1"
                                   oninput="updateRangeValue('umap-neighbors')">
                        </div>
                        
                        <div class="control-group">
                            <label for="umap-min-dist">
                                Min Distance: <span class="range-value" id="umap-min-dist-value">0.1</span>
                            </label>
                            <input type="range" id="umap-min-dist" min="0.0" max="1.0" value="0.1" step="0.05"
                                   oninput="updateRangeValue('umap-min-dist')">
                        </div>
                        
                        <div class="control-group">
                            <label for="umap-metric">Metric:</label>
                            <select id="umap-metric">
                                <option value="euclidean">Euclidean</option>
                                <option value="cosine">Cosine</option>
                                <option value="manhattan">Manhattan</option>
                            </select>
                        </div>
                    </div>
                    
                    <!-- PCA parameters -->
                    <div id="pca-params" class="algorithm-params">
                        <p style="color: #666; margin-top: 20px;">
                            PCA is a linear dimensionality reduction technique. 
                            No additional parameters needed.
                        </p>
                    </div>
                    
                    <!-- Clustering parameters -->
                    <div id="clustering-params" style="display: none; margin-top: 20px; padding: 15px; background-color: #f0f0f0; border-radius: 4px;">
                        <!-- K-Means parameters -->
                        <div id="kmeans-params" class="algorithm-params">
                            <h4 style="margin-top: 0;">K-Means Parameters</h4>
                            <div class="control-group">
                                <label for="kmeans-clusters">
                                    Number of Clusters: <span class="range-value" id="kmeans-clusters-value">5</span>
                                </label>
                                <input type="range" id="kmeans-clusters" min="2" max="20" value="5" step="1"
                                       oninput="updateRangeValue('kmeans-clusters')">
                            </div>
                        </div>
                        
                        <!-- DBSCAN parameters -->
                        <div id="dbscan-params" class="algorithm-params">
                            <h4 style="margin-top: 0;">DBSCAN Parameters</h4>
                            <div class="control-group">
                                <label for="dbscan-eps">
                                    Epsilon (neighborhood size): <span class="range-value" id="dbscan-eps-value">0.5</span>
                                </label>
                                <input type="range" id="dbscan-eps" min="0.1" max="5.0" value="0.5" step="0.1"
                                       oninput="updateRangeValue('dbscan-eps')">
                            </div>
                            <div class="control-group">
                                <label for="dbscan-min-samples">
                                    Min Samples: <span class="range-value" id="dbscan-min-samples-value">5</span>
                                </label>
                                <input type="range" id="dbscan-min-samples" min="2" max="20" value="5" step="1"
                                       oninput="updateRangeValue('dbscan-min-samples')">
                            </div>
                        </div>
                        
                        <!-- Hierarchical parameters -->
                        <div id="hierarchical-params" class="algorithm-params">
                            <h4 style="margin-top: 0;">Hierarchical Clustering Parameters</h4>
                            <div class="control-group">
                                <label for="hierarchical-clusters">
                                    Number of Clusters: <span class="range-value" id="hierarchical-clusters-value">5</span>
                                </label>
                                <input type="range" id="hierarchical-clusters" min="2" max="20" value="5" step="1"
                                       oninput="updateRangeValue('hierarchical-clusters')">
                            </div>
                            <div class="control-group">
                                <label for="hierarchical-linkage">Linkage:</label>
                                <select id="hierarchical-linkage">
                                    <option value="ward">Ward</option>
                                    <option value="complete">Complete</option>
                                    <option value="average">Average</option>
                                    <option value="single">Single</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    
                </div>
            </div>
            
            <button onclick="updateVisualization()" id="update-btn">Update Visualization</button>
            <button onclick="summarizeClusters()" id="summarize-btn" style="display: none; margin-left: 10px;">
                Summarize Clusters ✨
            </button>
            <button onclick="showSummaries()" id="show-summaries-btn">
                Show Summaries 📋
            </button>
            <button onclick="openDeleteClustersModal()" id="delete-clusters-btn" style="display: none; margin-left: 10px;">
                Delete Clusters 🗑️
            </button>
            <button onclick="openAnalyzeClusterModal()" id="analyze-cluster-btn" style="display: none; margin-left: 10px;">
                Analyze Cluster 🔍
            </button>
            
            <div id="status" class="status"></div>
            
            <!-- Breadcrumb navigation for recursive clustering -->
            <div id="breadcrumb" class="breadcrumb"></div>
            
            <!-- Drill-down info -->
            <div id="drill-down-info" class="drill-down-info">
                <strong>Drill-Down Mode:</strong> Click on any data point to explore its cluster. Each subcluster uses the same visualization settings as its parent. Use breadcrumb navigation above to go back.
            </div>
            
            <div class="config-controls">
                <label for="config-select">Configuration:</label>
                <select id="config-select" class="config-select" onchange="loadSelectedConfiguration()">
                    <option value="">-- Select Configuration --</option>
                </select>
                <button onclick="saveCurrentConfiguration()" class="save-config-btn">Save Config</button>
                <button id="delete-config-btn" onclick="deleteSelectedConfiguration()" class="delete-config-btn" style="display: none;">Delete</button>
                <button id="set-default-btn" onclick="toggleDefaultConfiguration()" style="display: none;">Set as Default</button>
            </div>
            
            <div class="config-controls" style="background-color: #e8f4f8; margin-top: 10px;">
                <label>Cluster State:</label>
                <select id="state-select" class="config-select" onchange="loadSelectedState()">
                    <option value="">-- Select State --</option>
                </select>
                <button onclick="saveCurrentState()" class="save-config-btn" style="background-color: #28a745;">Save State 💾</button>
                <button onclick="loadClusterStates()" class="save-config-btn" style="background-color: #6c757d;" title="Refresh state list">↻</button>
                <button id="delete-state-btn" onclick="deleteSelectedState()" class="delete-config-btn" style="display: none;">Delete</button>
                <button onclick="resetAllStates()" class="delete-config-btn" style="background-color: #dc3545;" title="Reset all saved states">Reset All</button>
            </div>
            
            <!-- Index Management Section -->
            <div class="config-controls" style="background-color: #f0e8ff; margin-top: 10px;">
                <label>Index Management:</label>
                <select id="index-select" class="config-select" onchange="loadSelectedIndex()">
                    <option value="">-- Select Index --</option>
                </select>
                <button onclick="showCreateIndexModal()" class="save-config-btn" style="background-color: #6f42c1;">Create New Index 📁</button>
                <button onclick="refreshIndices()" class="save-config-btn" style="background-color: #6c757d;" title="Refresh index list">↻</button>
                <button id="delete-index-btn" onclick="deleteSelectedIndex()" class="delete-config-btn" style="display: none;">Delete</button>
                <span id="index-info" style="margin-left: 10px; color: #666; font-size: 0.9em;"></span>
            </div>
            
            <!-- Search Section -->
            <div class="config-controls" style="background-color: #e8f8f5; margin-top: 10px;">
                <label>Search Documents:</label>
                <input type="text" id="search-query" placeholder="Enter search query..." style="flex: 1; margin-right: 10px;" onkeypress="if(event.key === 'Enter') searchDocuments()">
                <input type="number" id="search-top-k" value="10" min="1" max="50" style="width: 80px; margin-right: 10px;" title="Number of results" onkeypress="if(event.key === 'Enter') searchDocuments()">
                <button onclick="searchDocuments()" class="save-config-btn" style="background-color: #20c997;">Search 🔍</button>
                <button onclick="clearSearchResults()" class="delete-config-btn" style="background-color: #6c757d;">Clear</button>
            </div>
            
            <!-- Search Results Section -->
            <div id="search-results" style="display: none; margin-top: 20px; padding: 20px; background-color: #f8f9fa; border-radius: 8px; max-height: 400px; overflow-y: auto;">
                <h3 style="margin-top: 0; color: #333;">Search Results</h3>
                <div id="search-results-content"></div>
            </div>
            
            <!-- Create Index Modal -->
            <div id="create-index-modal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000;">
                <div style="position: relative; margin: 50px auto; width: 500px; background: white; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <button onclick="closeCreateIndexModal()" style="position: absolute; top: 10px; right: 10px; background: none; border: none; font-size: 24px; cursor: pointer; color: #666;">×</button>
                    <h2 style="margin-top: 0; color: #333;">Create New Index</h2>
                    
                    <div style="margin-bottom: 20px;">
                        <label for="new-index-name" style="display: block; margin-bottom: 5px;">Index Name:</label>
                        <input type="text" id="new-index-name" placeholder="Enter index name..." style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                    </div>
                    
                    <div style="margin-bottom: 20px;">
                        <label for="new-index-path" style="display: block; margin-bottom: 5px;">Folder Path:</label>
                        <input type="text" id="new-index-path" placeholder="/path/to/folder" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                    </div>
                    
                    <div id="indexing-progress" style="display: none; margin-bottom: 20px;">
                        <div style="background-color: #e8f4f8; padding: 10px; border-radius: 4px;">
                            <div style="margin-bottom: 5px;">Progress: <span id="indexing-progress-text">0/0</span></div>
                            <div style="background-color: #ddd; height: 20px; border-radius: 10px; overflow: hidden;">
                                <div id="indexing-progress-bar" style="background-color: #4CAF50; height: 100%; width: 0%; transition: width 0.3s;"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div id="indexing-error" style="display: none; margin-bottom: 20px; padding: 10px; background-color: #f8d7da; color: #721c24; border-radius: 4px;"></div>
                    
                    <div style="margin-top: 20px; border-top: 1px solid #eee; padding-top: 20px;">
                        <button onclick="createNewIndex()" style="background-color: #6f42c1; color: white;">Create Index</button>
                        <button onclick="closeCreateIndexModal()" style="margin-left: 10px;">Cancel</button>
                    </div>
                </div>
            </div>
            
            <!-- Cluster summaries modal -->
            <div id="summaries-modal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000;">
                <div style="position: relative; margin: 50px auto; width: 80%; max-width: 800px; max-height: 80vh; background: white; border-radius: 8px; padding: 20px; overflow-y: auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <button onclick="minimizeSummariesModal()" class="minimize-btn" title="Minimize">_</button>
                    <button onclick="closeSummariesModal()" style="position: absolute; top: 10px; right: 10px; background: none; border: none; font-size: 24px; cursor: pointer; color: #666;">×</button>
                    <h2 style="margin-top: 0; color: #333;">Cluster Summaries</h2>
                    <div id="delete-warning" class="delete-warning">
                        ⚠️ <strong>Warning:</strong> Deleting clusters will permanently remove the associated documents from the index.
                    </div>
                    <div id="summaries-content" style="min-height: 100px;"></div>
                    <div style="margin-top: 20px; border-top: 1px solid #eee; padding-top: 20px;">
                        <button onclick="exportSummaries('json')">Export as JSON</button>
                        <button onclick="exportSummaries('markdown')" style="margin-left: 10px;">Export as Markdown</button>
                        <button id="delete-selected-btn" class="delete-clusters-btn" onclick="deleteSelectedClusters()" style="display: none;">Delete Selected Clusters</button>
                    </div>
                </div>
            </div>
            
            <!-- Delete clusters modal -->
            <div id="delete-clusters-modal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000;">
                <div style="position: relative; margin: 50px auto; width: 600px; max-height: 80vh; background: white; border-radius: 8px; padding: 20px; overflow-y: auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <button onclick="closeDeleteClustersModal()" style="position: absolute; top: 10px; right: 10px; background: none; border: none; font-size: 24px; cursor: pointer; color: #666;">×</button>
                    <h2 style="margin-top: 0; color: #333;">Delete Clusters</h2>
                    <div class="delete-warning" style="display: block;">
                        ⚠️ <strong>Warning:</strong> Deleting clusters will permanently remove the associated documents from the index.
                    </div>
                    <div id="cluster-list" style="margin: 20px 0;"></div>
                    <div style="margin-top: 20px; border-top: 1px solid #eee; padding-top: 20px;">
                        <button id="delete-selected-clusters-btn" class="delete-clusters-btn" onclick="deleteSelectedClustersFromModal()" style="display: none;">Delete Selected Clusters</button>
                        <button onclick="closeDeleteClustersModal()" style="margin-left: 10px;">Cancel</button>
                    </div>
                </div>
            </div>
            
            <!-- Analyze cluster modal -->
            <div id="analyze-cluster-modal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000;">
                <div style="position: relative; margin: 50px auto; width: 600px; max-height: 90vh; background: white; border-radius: 8px; padding: 20px; overflow-y: auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <button onclick="closeAnalyzeClusterModal()" style="position: absolute; top: 10px; right: 10px; background: none; border: none; font-size: 24px; cursor: pointer; color: #666;">×</button>
                    <h2 style="margin-top: 0; color: #333;">Analyze Cluster</h2>
                    
                    <div style="margin-bottom: 20px;">
                        <h4>Select Cluster to Analyze:</h4>
                        <div id="analyze-cluster-list" style="margin: 10px 0; max-height: 200px; overflow-y: auto;"></div>
                    </div>
                    
                    <div style="margin-bottom: 20px; padding: 15px; background-color: #e8f4f8; border-radius: 4px;">
                        <p style="margin: 0; color: #666;">
                            <strong>Note:</strong> The selected cluster will be analyzed using the same visualization settings as the current view to maintain consistency.
                        </p>
                    </div>
                    
                    <div style="margin-top: 20px; border-top: 1px solid #eee; padding-top: 20px;">
                        <button onclick="analyzeSelectedCluster()" style="background-color: #17a2b8; color: white;">
                            Analyze Selected Cluster
                        </button>
                        <button onclick="closeAnalyzeClusterModal()" style="margin-left: 10px;">Cancel</button>
                    </div>
                </div>
            </div>
            
            <div id="plot"></div>
        </div>
        
        <script>
            function updateAlgorithmParams() {
                const algorithm = document.getElementById('algorithm').value;
                document.querySelectorAll('.algorithm-params').forEach(el => {
                    el.classList.remove('active');
                });
                document.getElementById(algorithm + '-params').classList.add('active');
            }
            
            function updateColorByParams() {
                const colorBy = document.getElementById('color_by').value;
                const clusteringParams = document.getElementById('clustering-params');
                
                if (['kmeans', 'dbscan', 'hierarchical'].includes(colorBy)) {
                    clusteringParams.style.display = 'block';
                    
                    // Hide all clustering params first
                    document.querySelectorAll('#clustering-params .algorithm-params').forEach(el => {
                        el.style.display = 'none';
                    });
                    
                    // Show relevant params
                    document.getElementById(colorBy + '-params').style.display = 'block';
                } else {
                    clusteringParams.style.display = 'none';
                }
            }
            
            function updateRangeValue(id) {
                const input = document.getElementById(id);
                const value = input.value;
                document.getElementById(id + '-value').textContent = value;
            }
            
            
            // Store current parameters globally
            let currentParams = {};
            let previousParams = {};
            let storedSummaries = null;
            let hierarchicalSummaries = {};  // Store summaries for each navigation level
            let savedConfigurations = {};
            let defaultConfig = null;
            let currentClusterPath = 'root';
            let clusterHistory = [];
            let currentClusterData = null;
            
            async function summarizeClusters() {
                const button = document.getElementById('summarize-btn');
                const status = document.getElementById('status');
                
                console.log('Summarizing clusters with params:', currentParams);
                console.log('Current cluster path:', currentClusterPath);
                
                button.disabled = true;
                status.className = 'status loading';
                status.textContent = 'Generating cluster summaries...';
                status.style.display = 'block';
                
                // Prepare request with cluster context
                const requestData = {
                    ...currentParams,
                    cluster_context: {
                        current_path: currentClusterPath
                    }
                };
                
                // If we're in a subcluster, add current cluster info
                if (currentClusterPath !== 'root' && clusterHistory.length > 0) {
                    const pathParts = currentClusterPath.split('-');
                    const parentClusterId = parseInt(pathParts[pathParts.length - 1]);
                    const parentIndex = clusterHistory.length - 1;
                    
                    if (parentIndex >= 0) {
                        requestData.cluster_context.parent_cluster_id = parentClusterId;
                        requestData.cluster_context.parent_params = clusterHistory[parentIndex].params;
                    }
                    
                    // Add current cluster's original indices if available
                    if (currentClusterData && currentClusterData.cluster_info && currentClusterData.cluster_info.original_indices) {
                        requestData.cluster_context.original_indices = currentClusterData.cluster_info.original_indices;
                        console.log('Sending original indices for drill-down cluster:', currentClusterData.cluster_info.original_indices.length);
                    }
                }
                
                try {
                    const response = await fetch('/api/summarize-clusters', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(requestData)
                    });
                    
                    console.log('Response status:', response.status);
                    const data = await response.json();
                    console.log('Response data:', data);
                    
                    if (response.ok) {
                        // Store summaries hierarchically
                        hierarchicalSummaries[currentClusterPath] = {
                            summaries: data.summaries,
                            timestamp: new Date().toISOString(),
                            params: {...currentParams}
                        };
                        
                        // Update the legacy storedSummaries for backward compatibility
                        storedSummaries = data.summaries;
                        
                        // Update the plot with cluster names
                        updatePlotWithClusterNames(data.summaries);
                        
                        displaySummaries(data.summaries);
                        status.className = 'status success';
                        status.textContent = 'Summaries generated successfully!';
                        document.getElementById('show-summaries-btn').style.display = 'inline-block';
                        setTimeout(() => {
                            status.style.display = 'none';
                        }, 3000);
                    } else {
                        throw new Error(data.error || 'Unknown error');
                    }
                } catch (error) {
                    console.error('Error in summarizeClusters:', error);
                    status.className = 'status error';
                    status.textContent = 'Error: ' + error.message;
                    status.style.display = 'block';
                } finally {
                    button.disabled = false;
                }
            }
            
            function displaySummaries(summaries) {
                console.log('Displaying summaries:', summaries);
                const modal = document.getElementById('summaries-modal');
                const content = document.getElementById('summaries-content');
                
                if (!summaries || Object.keys(summaries).length === 0) {
                    console.error('No summaries to display');
                    content.innerHTML = '<p>No cluster summaries available.</p>';
                    modal.style.display = 'block';
                    return;
                }
                
                // Clear previous content
                content.innerHTML = '';
                
                // Sort clusters by ID
                const sortedClusters = Object.keys(summaries).sort((a, b) => parseInt(a) - parseInt(b));
                console.log('Sorted cluster IDs:', sortedClusters);
                
                sortedClusters.forEach(clusterId => {
                    const summary = summaries[clusterId];
                    const div = document.createElement('div');
                    div.className = 'cluster-summary';
                    div.style.borderLeftColor = getClusterColor(parseInt(clusterId));
                    
                    // Use the generated name if available, otherwise fall back to "Cluster X"
                    const clusterTitle = summary.name && summary.name !== `Cluster ${clusterId}` 
                        ? `Cluster ${clusterId}: ${summary.name}` 
                        : `Cluster ${clusterId}`;
                    
                    div.innerHTML = `
                        <input type="checkbox" class="cluster-checkbox" value="${clusterId}" onchange="updateDeleteButton()">
                        <h3>🔍 ${clusterTitle} (${summary.doc_count} documents)</h3>
                        <div class="summary-text">${summary.summary}</div>
                        <div class="file-list">
                            <strong>Files:</strong> ${summary.files.join(', ')}
                        </div>
                    `;
                    
                    content.appendChild(div);
                });
                
                // Store summaries for export
                window.currentSummaries = summaries;
                
                // Show modal
                modal.style.display = 'block';
            }
            
            function updateDeleteButton() {
                const checkboxes = document.querySelectorAll('.cluster-checkbox:checked');
                const deleteBtn = document.getElementById('delete-selected-btn');
                const warning = document.getElementById('delete-warning');
                
                if (checkboxes.length > 0) {
                    deleteBtn.style.display = 'inline-block';
                    deleteBtn.textContent = `Delete ${checkboxes.length} Selected Cluster${checkboxes.length > 1 ? 's' : ''}`;
                    warning.style.display = 'block';
                } else {
                    deleteBtn.style.display = 'none';
                    warning.style.display = 'none';
                }
            }
            
            async function deleteSelectedClusters() {
                const checkboxes = document.querySelectorAll('.cluster-checkbox:checked');
                const clustersToDelete = Array.from(checkboxes).map(cb => parseInt(cb.value));
                
                if (clustersToDelete.length === 0) return;
                
                if (!confirm(`Are you sure you want to delete ${clustersToDelete.length} cluster(s)? This will permanently remove the associated documents from the index.`)) {
                    return;
                }
                
                const status = document.getElementById('status');
                status.className = 'status loading';
                status.textContent = 'Deleting selected clusters...';
                status.style.display = 'block';
                
                try {
                    const response = await fetch('/api/delete-clusters', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            clusters: clustersToDelete,
                            ...currentParams  // Include current visualization parameters
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        status.className = 'status success';
                        status.textContent = `Successfully deleted ${data.deleted_clusters.length} clusters (${data.documents_removed} documents removed)`;
                        
                        // Close modal and refresh visualization
                        closeSummariesModal();
                        
                        // Clear stored summaries since clusters were deleted
                        storedSummaries = null;
                        document.getElementById('show-summaries-btn').style.display = 'none';
                        
                        // Wait a moment then refresh
                        setTimeout(() => {
                            updateVisualization();
                            status.style.display = 'none';
                        }, 2000);
                    } else {
                        throw new Error(data.error || 'Unknown error');
                    }
                } catch (error) {
                    status.className = 'status error';
                    status.textContent = 'Error: ' + error.message;
                }
            }
            
            function closeSummariesModal() {
                document.getElementById('summaries-modal').style.display = 'none';
                // Clear checkboxes and reset delete button
                document.querySelectorAll('.cluster-checkbox').forEach(cb => cb.checked = false);
                updateDeleteButton();
            }
            
            function minimizeSummariesModal() {
                document.getElementById('summaries-modal').style.display = 'none';
                // Don't clear checkboxes when minimizing
            }
            
            function showSummaries() {
                // Check if we have summaries for the current navigation level
                const currentLevelSummaries = hierarchicalSummaries[currentClusterPath];
                
                if (currentLevelSummaries && currentLevelSummaries.summaries) {
                    // Update plot with cluster names before showing summaries
                    updatePlotWithClusterNames(currentLevelSummaries.summaries);
                    displaySummaries(currentLevelSummaries.summaries);
                } else if (storedSummaries && currentClusterPath === 'root') {
                    // Fallback to legacy storedSummaries for root level
                    updatePlotWithClusterNames(storedSummaries);
                    displaySummaries(storedSummaries);
                } else {
                    const status = document.getElementById('status');
                    status.className = 'status error';
                    status.textContent = `No summaries available for this ${currentClusterPath === 'root' ? 'level' : 'cluster'}. Please generate summaries first.`;
                    status.style.display = 'block';
                    setTimeout(() => {
                        status.style.display = 'none';
                    }, 3000);
                }
            }
            
            function getClusterColor(clusterId) {
                // Match colors from Plotly palette
                const colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
                               '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'];
                return colors[clusterId % colors.length];
            }
            
            function exportSummaries(format) {
                if (!window.currentSummaries) return;
                
                let content, filename, mimeType;
                
                if (format === 'json') {
                    content = JSON.stringify(window.currentSummaries, null, 2);
                    filename = 'cluster_summaries.json';
                    mimeType = 'application/json';
                } else if (format === 'markdown') {
                    content = '# Cluster Summaries\\n\\n';
                    const sortedClusters = Object.keys(window.currentSummaries).sort((a, b) => parseInt(a) - parseInt(b));
                    
                    sortedClusters.forEach(clusterId => {
                        const summary = window.currentSummaries[clusterId];
                        const clusterTitle = summary.name && summary.name !== `Cluster ${clusterId}` 
                            ? `Cluster ${clusterId}: ${summary.name}` 
                            : `Cluster ${clusterId}`;
                        content += `## ${clusterTitle} (${summary.doc_count} documents)\\n\\n`;
                        content += `${summary.summary}\\n\\n`;
                        content += `**Files:** ${summary.files.join(', ')}\\n\\n---\\n\\n`;
                    });
                    
                    filename = 'cluster_summaries.md';
                    mimeType = 'text/markdown';
                }
                
                // Create download
                const blob = new Blob([content], { type: mimeType });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
            
            // Override the updateVisualization function to store params and show/hide summarize button
            async function updateVisualizationWithParams() {
                // Store previous parameters
                previousParams = {...currentParams};
                
                // Store current parameters
                const algorithm = document.getElementById('algorithm').value;
                const dimensions = parseInt(document.getElementById('dimensions').value);
                const colorBy = document.getElementById('color_by').value;
                
                currentParams = {
                    algorithm: algorithm,
                    n_components: dimensions,
                    color_by: colorBy
                };
                
                // Add algorithm-specific parameters
                if (algorithm === 'tsne') {
                    currentParams.perplexity = parseInt(document.getElementById('tsne-perplexity').value);
                    currentParams.learning_rate = parseInt(document.getElementById('tsne-learning-rate').value);
                    currentParams.n_iter = parseInt(document.getElementById('tsne-iterations').value);
                    currentParams.early_exaggeration = parseInt(document.getElementById('tsne-early-exaggeration').value);
                } else if (algorithm === 'umap') {
                    currentParams.n_neighbors = parseInt(document.getElementById('umap-neighbors').value);
                    currentParams.min_dist = parseFloat(document.getElementById('umap-min-dist').value);
                    currentParams.metric = document.getElementById('umap-metric').value;
                }
                
                // Add clustering parameters if needed
                if (['kmeans', 'dbscan', 'hierarchical'].includes(colorBy)) {
                    currentParams.cluster_params = {};
                    
                    if (colorBy === 'kmeans') {
                        currentParams.cluster_params.n_clusters = parseInt(document.getElementById('kmeans-clusters').value);
                    } else if (colorBy === 'dbscan') {
                        currentParams.cluster_params.eps = parseFloat(document.getElementById('dbscan-eps').value);
                        currentParams.cluster_params.min_samples = parseInt(document.getElementById('dbscan-min-samples').value);
                    } else if (colorBy === 'hierarchical') {
                        currentParams.cluster_params.n_clusters = parseInt(document.getElementById('hierarchical-clusters').value);
                        currentParams.cluster_params.linkage = document.getElementById('hierarchical-linkage').value;
                    }
                }
                
                // Show/hide summarize, delete, and analyze buttons based on clustering
                const summarizeBtn = document.getElementById('summarize-btn');
                const showSummariesBtn = document.getElementById('show-summaries-btn');
                const deleteClustersBtn = document.getElementById('delete-clusters-btn');
                const analyzeClusterBtn = document.getElementById('analyze-cluster-btn');
                
                if (['kmeans', 'dbscan', 'hierarchical'].includes(colorBy)) {
                    summarizeBtn.style.display = 'inline-block';
                    deleteClustersBtn.style.display = 'inline-block';
                    analyzeClusterBtn.style.display = 'inline-block';
                    // Clear stored summaries when clustering parameters change
                    const paramsChanged = JSON.stringify(currentParams) !== JSON.stringify(previousParams);
                    if (paramsChanged) {
                        storedSummaries = null;
                        showSummariesBtn.style.display = 'none';
                    }
                } else {
                    summarizeBtn.style.display = 'none';
                    showSummariesBtn.style.display = 'none';
                    deleteClustersBtn.style.display = 'none';
                    analyzeClusterBtn.style.display = 'none';
                    storedSummaries = null;
                }
                
                // Now call the actual visualization update
                try {
                    const response = await fetch('/api/compute', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(currentParams)
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        // Update plot
                        const layout = data.plot.layout;
                        if (dimensions === 3) {
                            layout.scene = {
                                xaxis: {title: 'Component 1'},
                                yaxis: {title: 'Component 2'},
                                zaxis: {title: 'Component 3'}
                            };
                        }
                        
                        Plotly.newPlot('plot', data.plot.data, layout);
                        
                        // Add click handler for cluster drill-down
                        addPlotClickHandler();
                        
                        document.getElementById('status').className = 'status success';
                        document.getElementById('status').textContent = 'Visualization updated successfully!';
                        setTimeout(() => {
                            document.getElementById('status').style.display = 'none';
                        }, 3000);
                    } else {
                        throw new Error(data.error || 'Unknown error');
                    }
                } catch (error) {
                    document.getElementById('status').className = 'status error';
                    document.getElementById('status').textContent = 'Error: ' + error.message;
                } finally {
                    document.getElementById('update-btn').disabled = false;
                }
            }
            
            // Replace updateVisualization in onclick handlers
            window.updateVisualization = updateVisualizationWithParams;
            
            // Configuration management functions
            async function loadConfigurations() {
                try {
                    const response = await fetch('/api/configurations');
                    const data = await response.json();
                    
                    if (response.ok) {
                        savedConfigurations = data.configurations;
                        defaultConfig = data.default;
                        updateConfigurationDropdown();
                        
                        // Load default configuration if available
                        if (defaultConfig && savedConfigurations[defaultConfig]) {
                            loadConfiguration(savedConfigurations[defaultConfig]);
                        }
                    }
                } catch (error) {
                    console.error('Error loading configurations:', error);
                }
            }
            
            function updateConfigurationDropdown() {
                const select = document.getElementById('config-select');
                select.innerHTML = '<option value="">-- Select Configuration --</option>';
                
                Object.keys(savedConfigurations).forEach(name => {
                    const option = document.createElement('option');
                    option.value = name;
                    option.textContent = name;
                    if (name === defaultConfig) {
                        option.textContent += ' (Default)';
                    }
                    select.appendChild(option);
                });
            }
            
            function loadSelectedConfiguration() {
                const select = document.getElementById('config-select');
                const name = select.value;
                
                if (name && savedConfigurations[name]) {
                    loadConfiguration(savedConfigurations[name]);
                }
                
                // Update button visibility
                document.getElementById('delete-config-btn').style.display = name ? 'inline-block' : 'none';
                document.getElementById('set-default-btn').style.display = name ? 'inline-block' : 'none';
                
                // Update default button text
                if (name) {
                    const btn = document.getElementById('set-default-btn');
                    btn.textContent = name === defaultConfig ? 'Remove as Default' : 'Set as Default';
                }
            }
            
            function loadConfiguration(config) {
                // Load algorithm
                if (config.algorithm) {
                    document.getElementById('algorithm').value = config.algorithm;
                    updateAlgorithmParams();
                }
                
                // Load dimensions
                if (config.n_components) {
                    document.getElementById('dimensions').value = config.n_components;
                }
                
                // Load color by
                if (config.color_by) {
                    document.getElementById('color_by').value = config.color_by;
                    updateColorByParams();
                }
                
                // Load algorithm-specific parameters
                if (config.algorithm === 'tsne') {
                    if (config.perplexity) {
                        document.getElementById('tsne-perplexity').value = config.perplexity;
                        updateRangeValue('tsne-perplexity');
                    }
                    if (config.learning_rate) {
                        document.getElementById('tsne-learning-rate').value = config.learning_rate;
                        updateRangeValue('tsne-learning-rate');
                    }
                    if (config.n_iter) {
                        document.getElementById('tsne-iterations').value = config.n_iter;
                        updateRangeValue('tsne-iterations');
                    }
                    if (config.early_exaggeration) {
                        document.getElementById('tsne-early-exaggeration').value = config.early_exaggeration;
                        updateRangeValue('tsne-early-exaggeration');
                    }
                } else if (config.algorithm === 'umap') {
                    if (config.n_neighbors) {
                        document.getElementById('umap-neighbors').value = config.n_neighbors;
                        updateRangeValue('umap-neighbors');
                    }
                    if (config.min_dist) {
                        document.getElementById('umap-min-dist').value = config.min_dist;
                        updateRangeValue('umap-min-dist');
                    }
                    if (config.metric) {
                        document.getElementById('umap-metric').value = config.metric;
                    }
                }
                
                // Load clustering parameters
                if (config.cluster_params) {
                    if (config.color_by === 'kmeans' && config.cluster_params.n_clusters) {
                        document.getElementById('kmeans-clusters').value = config.cluster_params.n_clusters;
                        updateRangeValue('kmeans-clusters');
                    } else if (config.color_by === 'dbscan') {
                        if (config.cluster_params.eps) {
                            document.getElementById('dbscan-eps').value = config.cluster_params.eps;
                            updateRangeValue('dbscan-eps');
                        }
                        if (config.cluster_params.min_samples) {
                            document.getElementById('dbscan-min-samples').value = config.cluster_params.min_samples;
                            updateRangeValue('dbscan-min-samples');
                        }
                    } else if (config.color_by === 'hierarchical') {
                        if (config.cluster_params.n_clusters) {
                            document.getElementById('hierarchical-clusters').value = config.cluster_params.n_clusters;
                            updateRangeValue('hierarchical-clusters');
                        }
                        if (config.cluster_params.linkage) {
                            document.getElementById('hierarchical-linkage').value = config.cluster_params.linkage;
                        }
                    }
                }
                
                // Update visualization
                updateVisualizationWithParams();
            }
            
            async function saveCurrentConfiguration() {
                const name = prompt('Enter a name for this configuration:');
                if (!name) return;
                
                const setAsDefault = confirm('Set this as the default configuration?');
                
                try {
                    const response = await fetch('/api/configurations', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            name: name,
                            params: currentParams,
                            setAsDefault: setAsDefault
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        const status = document.getElementById('status');
                        status.className = 'status success';
                        status.textContent = data.message;
                        status.style.display = 'block';
                        
                        // Reload configurations
                        await loadConfigurations();
                        
                        // Select the new configuration
                        document.getElementById('config-select').value = name;
                        loadSelectedConfiguration();
                        
                        setTimeout(() => {
                            status.style.display = 'none';
                        }, 3000);
                    } else {
                        throw new Error(data.error || 'Unknown error');
                    }
                } catch (error) {
                    const status = document.getElementById('status');
                    status.className = 'status error';
                    status.textContent = 'Error: ' + error.message;
                    status.style.display = 'block';
                }
            }
            
            async function deleteSelectedConfiguration() {
                const select = document.getElementById('config-select');
                const name = select.value;
                
                if (!name) return;
                
                if (!confirm(`Are you sure you want to delete the configuration "${name}"?`)) {
                    return;
                }
                
                try {
                    const response = await fetch(`/api/configurations/${encodeURIComponent(name)}`, {
                        method: 'DELETE'
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        const status = document.getElementById('status');
                        status.className = 'status success';
                        status.textContent = data.message;
                        status.style.display = 'block';
                        
                        // Reload configurations
                        await loadConfigurations();
                        
                        setTimeout(() => {
                            status.style.display = 'none';
                        }, 3000);
                    } else {
                        throw new Error(data.error || 'Unknown error');
                    }
                } catch (error) {
                    const status = document.getElementById('status');
                    status.className = 'status error';
                    status.textContent = 'Error: ' + error.message;
                    status.style.display = 'block';
                }
            }
            
            async function toggleDefaultConfiguration() {
                const select = document.getElementById('config-select');
                const name = select.value;
                
                if (!name) return;
                
                const isCurrentDefault = name === defaultConfig;
                const newDefault = isCurrentDefault ? null : name;
                
                try {
                    const response = await fetch('/api/configurations/default', {
                        method: 'PUT',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            name: newDefault
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        const status = document.getElementById('status');
                        status.className = 'status success';
                        status.textContent = data.message;
                        status.style.display = 'block';
                        
                        // Reload configurations
                        await loadConfigurations();
                        
                        // Reselect current configuration
                        select.value = name;
                        loadSelectedConfiguration();
                        
                        setTimeout(() => {
                            status.style.display = 'none';
                        }, 3000);
                    } else {
                        throw new Error(data.error || 'Unknown error');
                    }
                } catch (error) {
                    const status = document.getElementById('status');
                    status.className = 'status error';
                    status.textContent = 'Error: ' + error.message;
                    status.style.display = 'block';
                }
            }
            
            // Delete clusters modal functions
            async function openDeleteClustersModal() {
                const modal = document.getElementById('delete-clusters-modal');
                const clusterList = document.getElementById('cluster-list');
                
                // Clear previous content
                clusterList.innerHTML = '<p style="text-align: center;">Loading clusters...</p>';
                modal.style.display = 'block';
                
                try {
                    // Get current cluster information
                    const response = await fetch('/api/compute', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(currentParams)
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        // Extract cluster information from plot data
                        const clusters = {};
                        data.plot.data.forEach(trace => {
                            if (trace.name && trace.name.startsWith('Cluster ')) {
                                const clusterId = parseInt(trace.name.replace('Cluster ', ''));
                                clusters[clusterId] = {
                                    id: clusterId,
                                    count: trace.x.length,
                                    name: trace.name
                                };
                            } else if (trace.name === 'Outliers') {
                                clusters[-1] = {
                                    id: -1,
                                    count: trace.x.length,
                                    name: 'Outliers'
                                };
                            }
                        });
                        
                        // Display clusters
                        clusterList.innerHTML = '';
                        const sortedClusters = Object.values(clusters).sort((a, b) => a.id - b.id);
                        
                        if (sortedClusters.length === 0) {
                            clusterList.innerHTML = '<p>No clusters found. Make sure clustering is enabled.</p>';
                            return;
                        }
                        
                        sortedClusters.forEach(cluster => {
                            const div = document.createElement('div');
                            div.className = 'cluster-item';
                            div.style.borderLeftColor = getClusterColor(cluster.id);
                            div.style.borderLeftWidth = '4px';
                            
                            div.innerHTML = `
                                <div style="display: flex; align-items: center;">
                                    <input type="checkbox" value="${cluster.id}" onchange="updateDeleteClustersButton()">
                                    <span><strong>${cluster.name}</strong></span>
                                </div>
                                <span class="cluster-stats">${cluster.count} documents</span>
                            `;
                            
                            clusterList.appendChild(div);
                        });
                    } else {
                        throw new Error(data.error || 'Failed to load clusters');
                    }
                } catch (error) {
                    clusterList.innerHTML = `<p style="color: red;">Error loading clusters: ${error.message}</p>`;
                }
            }
            
            function closeDeleteClustersModal() {
                document.getElementById('delete-clusters-modal').style.display = 'none';
                // Clear checkboxes
                document.querySelectorAll('#cluster-list input[type="checkbox"]').forEach(cb => cb.checked = false);
                updateDeleteClustersButton();
            }
            
            function updateDeleteClustersButton() {
                const checkboxes = document.querySelectorAll('#cluster-list input[type="checkbox"]:checked');
                const deleteBtn = document.getElementById('delete-selected-clusters-btn');
                
                if (checkboxes.length > 0) {
                    deleteBtn.style.display = 'inline-block';
                    deleteBtn.textContent = `Delete ${checkboxes.length} Selected Cluster${checkboxes.length > 1 ? 's' : ''}`;
                } else {
                    deleteBtn.style.display = 'none';
                }
            }
            
            async function deleteSelectedClustersFromModal() {
                const checkboxes = document.querySelectorAll('#cluster-list input[type="checkbox"]:checked');
                const clustersToDelete = Array.from(checkboxes).map(cb => parseInt(cb.value));
                
                if (clustersToDelete.length === 0) return;
                
                if (!confirm(`Are you sure you want to delete ${clustersToDelete.length} cluster(s)? This will permanently remove the associated documents from the index.`)) {
                    return;
                }
                
                const status = document.getElementById('status');
                status.className = 'status loading';
                status.textContent = 'Deleting selected clusters...';
                status.style.display = 'block';
                
                try {
                    const response = await fetch('/api/delete-clusters', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            clusters: clustersToDelete,
                            ...currentParams
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        status.className = 'status success';
                        status.textContent = `Successfully deleted ${data.deleted_clusters.length} clusters (${data.documents_removed} documents removed)`;
                        
                        // Close modal
                        closeDeleteClustersModal();
                        
                        // Clear stored summaries since clusters were deleted
                        storedSummaries = null;
                        document.getElementById('show-summaries-btn').style.display = 'none';
                        
                        // Wait a moment then refresh
                        setTimeout(() => {
                            updateVisualization();
                            status.style.display = 'none';
                        }, 2000);
                    } else {
                        throw new Error(data.error || 'Unknown error');
                    }
                } catch (error) {
                    status.className = 'status error';
                    status.textContent = 'Error: ' + error.message;
                }
            }
            
            // Analyze cluster functions
            async function openAnalyzeClusterModal() {
                const modal = document.getElementById('analyze-cluster-modal');
                const clusterList = document.getElementById('analyze-cluster-list');
                
                // Clear previous content
                clusterList.innerHTML = '<p style="text-align: center;">Loading clusters...</p>';
                modal.style.display = 'block';
                
                // No need to initialize parameters - we'll use parent settings
                
                try {
                    // Get current cluster information
                    const response = await fetch('/api/compute', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(currentParams)
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        // Extract cluster information from plot data
                        const clusters = {};
                        data.plot.data.forEach(trace => {
                            if (trace.name && trace.name.startsWith('Cluster ')) {
                                const clusterId = parseInt(trace.name.replace('Cluster ', ''));
                                clusters[clusterId] = {
                                    id: clusterId,
                                    count: trace.x.length,
                                    name: trace.name
                                };
                            } else if (trace.name === 'Outliers') {
                                clusters[-1] = {
                                    id: -1,
                                    count: trace.x.length,
                                    name: 'Outliers'
                                };
                            }
                        });
                        
                        // Display clusters with radio buttons (single selection)
                        clusterList.innerHTML = '';
                        const sortedClusters = Object.values(clusters).sort((a, b) => a.id - b.id);
                        
                        if (sortedClusters.length === 0) {
                            clusterList.innerHTML = '<p>No clusters found. Make sure clustering is enabled.</p>';
                            return;
                        }
                        
                        sortedClusters.forEach((cluster, index) => {
                            const div = document.createElement('div');
                            div.className = 'cluster-item';
                            div.style.borderLeftColor = getClusterColor(cluster.id);
                            div.style.borderLeftWidth = '4px';
                            
                            div.innerHTML = `
                                <div style="display: flex; align-items: center;">
                                    <input type="radio" name="analyze-cluster" value="${cluster.id}" ${index === 0 ? 'checked' : ''}>
                                    <span style="margin-left: 10px;"><strong>${cluster.name}</strong></span>
                                </div>
                                <span class="cluster-stats">${cluster.count} documents</span>
                            `;
                            
                            clusterList.appendChild(div);
                        });
                    } else {
                        throw new Error(data.error || 'Failed to load clusters');
                    }
                } catch (error) {
                    clusterList.innerHTML = `<p style="color: red;">Error loading clusters: ${error.message}</p>`;
                }
            }
            
            function closeAnalyzeClusterModal() {
                document.getElementById('analyze-cluster-modal').style.display = 'none';
            }
            
            // These functions are no longer needed since we inherit parent settings
            
            async function analyzeSelectedCluster() {
                const selectedCluster = document.querySelector('input[name="analyze-cluster"]:checked');
                if (!selectedCluster) {
                    alert('Please select a cluster to analyze');
                    return;
                }
                
                const clusterId = parseInt(selectedCluster.value);
                
                // Close modal
                closeAnalyzeClusterModal();
                
                // Show loading status
                const status = document.getElementById('status');
                status.className = 'status loading';
                status.textContent = `Analyzing cluster ${clusterId} using parent configuration...`;
                status.style.display = 'block';
                
                try {
                    const response = await fetch('/api/analyze-cluster', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            current_params: currentParams,
                            cluster_id: clusterId,
                            current_path: currentClusterPath
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        // Store cluster history with FULL visualization data
                        clusterHistory.push({
                            path: currentClusterPath,
                            params: {...currentParams},
                            data: currentClusterData,
                            plot: currentClusterData ? currentClusterData.plot : null
                        });
                        
                        // Update current state with new path from server
                        currentClusterPath = data.new_path;
                        currentClusterData = data;
                        
                        // Store this visualization in our navigation tree
                        if (!window.navigationTree) {
                            window.navigationTree = {};
                        }
                        window.navigationTree[currentClusterPath] = {
                            plot: data.plot,
                            breadcrumbs: data.breadcrumbs,
                            cluster_info: data.cluster_info,
                            visualization_params: data.visualization_params,
                            parent_params: data.parent_params,
                            timestamp: new Date().toISOString()
                        };
                        
                        // Update breadcrumbs
                        updateBreadcrumbs(data.breadcrumbs);
                        
                        // Enable drill-down mode
                        document.getElementById('drill-down-info').classList.add('active');
                        document.getElementById('breadcrumb').classList.add('active');
                        
                        // Update plot
                        Plotly.newPlot('plot', data.plot.data, data.plot.layout);
                        
                        // Add click handler for cluster drill-down
                        addPlotClickHandler();
                        
                        // Update status
                        status.className = 'status success';
                        status.textContent = `Successfully analyzed cluster ${clusterId}`;
                        
                        // Check if we have summaries for this new level
                        updateShowSummariesButton();
                        
                        setTimeout(() => {
                            status.style.display = 'none';
                        }, 3000);
                    } else {
                        throw new Error(data.error || 'Unknown error');
                    }
                } catch (error) {
                    status.className = 'status error';
                    status.textContent = 'Error: ' + error.message;
                }
            }
            
            function updateBreadcrumbs(breadcrumbs) {
                const container = document.getElementById('breadcrumb');
                container.innerHTML = '';
                
                breadcrumbs.forEach((crumb, index) => {
                    if (index > 0) {
                        const separator = document.createElement('span');
                        separator.className = 'breadcrumb-separator';
                        separator.textContent = ' > ';
                        container.appendChild(separator);
                    }
                    
                    const link = document.createElement('a');
                    link.className = 'breadcrumb-item';
                    link.textContent = crumb.label;
                    link.onclick = () => navigateToCluster(crumb.id);
                    
                    container.appendChild(link);
                });
            }
            
            async function navigateToCluster(path) {
                if (path === currentClusterPath) return;
                
                if (path === 'root') {
                    // Go back to root
                    currentClusterPath = 'root';
                    clusterHistory = [];
                    currentClusterData = null;
                    
                    // Hide breadcrumbs
                    document.getElementById('drill-down-info').classList.remove('active');
                    document.getElementById('breadcrumb').classList.remove('active');
                    
                    // Check if we have summaries for root level
                    updateShowSummariesButton();
                    
                    // Reload original visualization
                    updateVisualization();
                } else {
                    // First check if we have this path in our navigation tree
                    if (window.navigationTree && window.navigationTree[path]) {
                        const storedData = window.navigationTree[path];
                        currentClusterPath = path;
                        currentClusterData = storedData;
                        
                        // Rebuild history up to this point
                        clusterHistory = [];
                        const pathParts = path.split('-');
                        for (let i = 0; i < pathParts.length; i++) {
                            const subPath = pathParts.slice(0, i + 1).join('-');
                            if (window.navigationTree[subPath]) {
                                clusterHistory.push({
                                    path: i === 0 ? 'root' : pathParts.slice(0, i).join('-'),
                                    params: window.navigationTree[subPath].parent_params || currentParams,
                                    data: window.navigationTree[subPath],
                                    plot: window.navigationTree[subPath].plot
                                });
                            }
                        }
                        
                        // Update plot with stored data
                        Plotly.newPlot('plot', storedData.plot.data, storedData.plot.layout);
                        updateBreadcrumbs(storedData.breadcrumbs);
                        
                        // Add click handler for cluster drill-down
                        addPlotClickHandler();
                        
                        // Make sure drill-down UI is active
                        document.getElementById('drill-down-info').classList.add('active');
                        document.getElementById('breadcrumb').classList.add('active');
                    } else {
                        // Fallback to old history-based navigation
                        const pathParts = path.split('-');
                        const historyIndex = pathParts.length - 1;
                        
                        if (historyIndex < clusterHistory.length) {
                            const historyItem = clusterHistory[historyIndex - 1];
                            currentClusterPath = path;
                            currentClusterData = historyItem.data;
                            clusterHistory = clusterHistory.slice(0, historyIndex);
                            
                            if (currentClusterData && currentClusterData.plot) {
                                Plotly.newPlot('plot', currentClusterData.plot.data, currentClusterData.plot.layout);
                                updateBreadcrumbs(currentClusterData.breadcrumbs);
                                
                                // Add click handler for cluster drill-down
                                addPlotClickHandler();
                            }
                        }
                    }
                    
                    // Check if we have summaries for this level
                    updateShowSummariesButton();
                    
                    // Apply cluster names if summaries exist for this level
                    const levelSummaries = hierarchicalSummaries[path];
                    if (levelSummaries && levelSummaries.summaries) {
                        updatePlotWithClusterNames(levelSummaries.summaries);
                    }
                }
            }
            
            function addPlotClickHandler() {
                // Add click event listener to the plot
                const plotDiv = document.getElementById('plot');
                
                plotDiv.on('plotly_click', function(data) {
                    // Check if clustering is active
                    const colorBy = currentParams.color_by;
                    if (!['kmeans', 'dbscan', 'hierarchical'].includes(colorBy)) {
                        return; // No clustering active, ignore clicks
                    }
                    
                    // Get the cluster label from the clicked point
                    const pointData = data.points[0];
                    const clusterId = pointData.data.customdata ? pointData.data.customdata[0] : null;
                    
                    if (clusterId !== null && clusterId !== undefined && clusterId !== -1) {
                        // -1 is typically noise in DBSCAN, don't drill down
                        
                        // Show loading status
                        const status = document.getElementById('status');
                        status.className = 'status loading';
                        status.textContent = `Drilling down into cluster ${clusterId}...`;
                        status.style.display = 'block';
                        
                        // Directly analyze the cluster without modal
                        fetch('/api/analyze-cluster', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                current_params: currentParams,
                                cluster_id: clusterId,
                                current_path: currentClusterPath
                            })
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'success') {
                                // Store cluster history with FULL visualization data
                                clusterHistory.push({
                                    path: currentClusterPath,
                                    params: {...currentParams},
                                    data: currentClusterData,
                                    plot: currentClusterData ? currentClusterData.plot : null
                                });
                                
                                // Update current state with new path from server
                                currentClusterPath = data.new_path;
                                currentClusterData = data;
                                
                                // Store this visualization in our navigation tree
                                if (!window.navigationTree) {
                                    window.navigationTree = {};
                                }
                                window.navigationTree[currentClusterPath] = {
                                    plot: data.plot,
                                    breadcrumbs: data.breadcrumbs,
                                    cluster_info: data.cluster_info,
                                    visualization_params: data.visualization_params,
                                    parent_params: data.parent_params,
                                    timestamp: new Date().toISOString()
                                };
                                
                                // Update breadcrumbs
                                updateBreadcrumbs(data.breadcrumbs);
                                
                                // Enable drill-down mode
                                document.getElementById('drill-down-info').classList.add('active');
                                document.getElementById('breadcrumb').classList.add('active');
                                
                                // Update plot
                                Plotly.newPlot('plot', data.plot.data, data.plot.layout);
                                
                                // Add click handler to new plot
                                addPlotClickHandler();
                                
                                // Update status
                                status.className = 'status success';
                                status.textContent = `Successfully drilled down into cluster ${clusterId}`;
                                
                                // Check if we have summaries for this new level
                                updateShowSummariesButton();
                                
                                setTimeout(() => {
                                    status.style.display = 'none';
                                }, 3000);
                            } else {
                                throw new Error(data.error || 'Unknown error');
                            }
                        })
                        .catch(error => {
                            status.className = 'status error';
                            status.textContent = 'Error: ' + error.message;
                        });
                    }
                });
            }
            
            function updateShowSummariesButton() {
                const showSummariesBtn = document.getElementById('show-summaries-btn');
                const hasCurrentLevelSummaries = hierarchicalSummaries[currentClusterPath] && hierarchicalSummaries[currentClusterPath].summaries;
                const hasRootSummaries = storedSummaries && currentClusterPath === 'root';
                
                if (hasCurrentLevelSummaries || hasRootSummaries) {
                    showSummariesBtn.style.display = 'inline-block';
                } else {
                    showSummariesBtn.style.display = 'none';
                }
            }
            
            function updatePlotWithClusterNames(summaries) {
                // Get current plot data
                const plotDiv = document.getElementById('plot');
                if (!plotDiv || !plotDiv.data) return;
                
                const currentData = [...plotDiv.data];  // Clone the data array
                let updated = false;
                
                // Update trace names and hover text with cluster names
                currentData.forEach(trace => {
                    if (trace.name && trace.name.startsWith('Cluster ')) {
                        // Extract cluster ID, handling names that might already have been updated
                        const match = trace.name.match(/Cluster (\\d+)/);
                        if (match) {
                            const clusterId = parseInt(match[1]);
                            if (summaries[clusterId] && summaries[clusterId].name && summaries[clusterId].name !== `Cluster ${clusterId}`) {
                                const newName = `Cluster ${clusterId}: ${summaries[clusterId].name}`;
                                
                                // Update trace name (legend)
                                trace.name = newName;
                                
                                // Update hover text for all points in this trace
                                if (trace.text && Array.isArray(trace.text)) {
                                    trace.text = trace.text.map(text => {
                                        // Replace "Cluster: Cluster X" with "Cluster: Cluster X: Name"
                                        return text.replace(
                                            /Cluster: Cluster \\d+/,
                                            `Cluster: ${newName}`
                                        );
                                    });
                                }
                                
                                updated = true;
                            }
                        }
                    }
                });
                
                // Update the plot with new names if any changes were made
                if (updated) {
                    Plotly.react('plot', currentData, plotDiv.layout);
                }
            }
            
            // Cluster state management functions
            let savedStates = {};
            let lastUsedState = null;
            
            async function loadClusterStates() {
                try {
                    const response = await fetch('/api/cluster-states');
                    const data = await response.json();
                    
                    if (response.ok) {
                        savedStates = data.states;
                        lastUsedState = data.last_used;
                        updateStateSelect();
                        return data;
                    }
                } catch (error) {
                    console.error('Error loading cluster states:', error);
                }
                return null;
            }
            
            function updateStateSelect() {
                const select = document.getElementById('state-select');
                select.innerHTML = '<option value="">-- Select State --</option>';
                
                Object.keys(savedStates).forEach(name => {
                    const state = savedStates[name];
                    const option = document.createElement('option');
                    option.value = name;
                    option.textContent = `${name} (${state.saved_at})`;
                    select.appendChild(option);
                });
                
                // Show/hide delete button
                document.getElementById('delete-state-btn').style.display = 
                    select.value && select.value !== '' ? 'inline-block' : 'none';
            }
            
            async function saveCurrentState() {
                const name = prompt('Enter a name for this cluster state:');
                if (!name) return;
                
                const description = prompt('Enter a description (optional):') || '';
                
                const status = document.getElementById('status');
                status.className = 'status loading';
                status.textContent = 'Saving cluster state...';
                status.style.display = 'block';
                
                try {
                    const stateData = {
                        name: name,
                        description: description,
                        visualization_params: currentParams,
                        hierarchical_summaries: hierarchicalSummaries,
                        cluster_history: clusterHistory,
                        current_path: currentClusterPath,
                        current_data: currentClusterData,
                        navigation_tree: window.navigationTree || {}  // Include full navigation tree
                    };
                    
                    const response = await fetch('/api/cluster-states', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(stateData)
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        status.className = 'status success';
                        status.textContent = data.message;
                        
                        // Reload states
                        await loadClusterStates();
                        
                        // Select the newly saved state
                        document.getElementById('state-select').value = name;
                        
                        setTimeout(() => {
                            status.style.display = 'none';
                        }, 3000);
                    } else {
                        throw new Error(data.error || 'Unknown error');
                    }
                } catch (error) {
                    status.className = 'status error';
                    status.textContent = 'Error: ' + error.message;
                }
            }
            
            async function loadSelectedState(autoLoad = false) {
                const select = document.getElementById('state-select');
                const name = select.value;
                
                if (!name) {
                    document.getElementById('delete-state-btn').style.display = 'none';
                    return;
                }
                
                document.getElementById('delete-state-btn').style.display = 'inline-block';
                
                // Skip confirmation for auto-load
                if (!autoLoad && !confirm(`Load cluster state "${name}"? This will replace the current visualization and summaries.`)) {
                    return;
                }
                
                const status = document.getElementById('status');
                status.className = 'status loading';
                status.textContent = 'Loading cluster state...';
                status.style.display = 'block';
                
                try {
                    const response = await fetch(`/api/cluster-states/${encodeURIComponent(name)}`);
                    const data = await response.json();
                    
                    if (response.ok) {
                        const state = data.state;
                        
                        // Check for validation warnings
                        if (data.validation_warnings && data.validation_warnings.length > 0) {
                            const warningMessage = 'The loaded state had some issues that were automatically fixed:\\n\\n' + 
                                                 data.validation_warnings.join('\\n') + 
                                                 '\\n\\nThe state has been cleaned and will work properly.';
                            alert(warningMessage);
                        }
                        
                        // Restore state
                        currentParams = state.visualization_params || {};
                        hierarchicalSummaries = state.hierarchical_summaries || {};
                        clusterHistory = state.cluster_history || [];
                        currentClusterPath = state.current_path || 'root';
                        currentClusterData = state.current_data;
                        
                        // Restore navigation tree for subcluster navigation
                        if (state.navigation_tree) {
                            window.navigationTree = state.navigation_tree;
                        }
                        
                        // Update UI to reflect loaded parameters
                        if (currentParams.algorithm) {
                            document.getElementById('algorithm').value = currentParams.algorithm;
                            updateAlgorithmParams();
                        }
                        if (currentParams.n_components) {
                            document.getElementById('dimensions').value = currentParams.n_components;
                        }
                        if (currentParams.color_by) {
                            document.getElementById('color_by').value = currentParams.color_by;
                            updateColorByParams();
                        }
                        
                        // Update summaries button visibility
                        updateShowSummariesButton();
                        
                        // Apply cluster names if summaries exist for current level
                        const currentSummaries = hierarchicalSummaries[currentClusterPath];
                        if (currentSummaries && currentSummaries.summaries) {
                            // Delay slightly to ensure plot is rendered first
                            setTimeout(() => {
                                updatePlotWithClusterNames(currentSummaries.summaries);
                            }, 100);
                        }
                        
                        // If we're in a subcluster, restore the view
                        if (currentClusterPath !== 'root' && currentClusterData) {
                            // Update breadcrumbs
                            if (currentClusterData.breadcrumbs) {
                                updateBreadcrumbs(currentClusterData.breadcrumbs);
                            }
                            
                            // Enable drill-down mode
                            document.getElementById('drill-down-info').classList.add('active');
                            document.getElementById('breadcrumb').classList.add('active');
                            
                            // Update plot
                            if (currentClusterData.plot) {
                                Plotly.newPlot('plot', currentClusterData.plot.data, currentClusterData.plot.layout);
                            }
                        } else {
                            // Load root visualization
                            updateVisualization();
                        }
                        
                        status.className = 'status success';
                        status.textContent = `Loaded cluster state "${name}"`;
                        
                        setTimeout(() => {
                            status.style.display = 'none';
                        }, 3000);
                    } else {
                        throw new Error(data.error || 'Unknown error');
                    }
                } catch (error) {
                    status.className = 'status error';
                    status.textContent = 'Error: ' + error.message;
                }
            }
            
            async function deleteSelectedState() {
                const select = document.getElementById('state-select');
                const name = select.value;
                
                if (!name) return;
                
                if (!confirm(`Are you sure you want to delete the cluster state "${name}"? This action cannot be undone.`)) {
                    return;
                }
                
                const status = document.getElementById('status');
                
                try {
                    const response = await fetch(`/api/cluster-states/${encodeURIComponent(name)}`, {
                        method: 'DELETE'
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        status.className = 'status success';
                        status.textContent = data.message;
                        status.style.display = 'block';
                        
                        // Reload states
                        await loadClusterStates();
                        
                        setTimeout(() => {
                            status.style.display = 'none';
                        }, 3000);
                    } else {
                        throw new Error(data.error || 'Unknown error');
                    }
                } catch (error) {
                    status.className = 'status error';
                    status.textContent = 'Error: ' + error.message;
                    status.style.display = 'block';
                }
            }
            
            async function resetAllStates() {
                const message = 'Are you sure you want to reset ALL saved states? This will:\\n\\n' +
                               '• Delete all saved cluster states\\n' +
                               '• Clear navigation history\\n' +
                               '• Reset to default visualization\\n\\n' +
                               'This action cannot be undone.';
                
                if (!confirm(message)) {
                    return;
                }
                
                const resetConfigs = confirm('Also reset saved configurations?');
                
                const status = document.getElementById('status');
                status.className = 'status loading';
                status.textContent = 'Resetting all states...';
                status.style.display = 'block';
                
                try {
                    const response = await fetch('/api/reset-states', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            reset_configurations: resetConfigs
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        // Reset local state
                        savedStates = {};
                        hierarchicalSummaries = {};
                        clusterHistory = [];
                        currentClusterPath = 'root';
                        currentClusterData = null;
                        window.navigationTree = {};
                        
                        // Reset UI
                        document.getElementById('state-select').innerHTML = '<option value="">-- Select State --</option>';
                        document.getElementById('delete-state-btn').style.display = 'none';
                        
                        // Hide breadcrumbs and drill-down info
                        document.getElementById('drill-down-info').classList.remove('active');
                        document.getElementById('breadcrumb').classList.remove('active');
                        
                        status.className = 'status success';
                        status.textContent = data.message;
                        
                        // Reload configurations if they were reset
                        if (resetConfigs) {
                            savedConfigurations = {};
                            defaultConfig = null;
                            updateConfigurationDropdown();
                        }
                        
                        // Reload default visualization
                        updateVisualizationWithParams();
                        
                        setTimeout(() => {
                            status.style.display = 'none';
                        }, 3000);
                    } else {
                        throw new Error(data.error || 'Unknown error');
                    }
                } catch (error) {
                    status.className = 'status error';
                    status.textContent = 'Error: ' + error.message;
                }
            }
            
            // Index Management Functions
            let currentIndexName = null;
            let indexingTaskId = null;
            let indexingInterval = null;
            
            async function loadIndices() {
                try {
                    const response = await fetch('/api/indices');
                    const data = await response.json();
                    
                    if (response.ok) {
                        const select = document.getElementById('index-select');
                        select.innerHTML = '<option value="">-- Select Index --</option>';
                        
                        // Add indices to dropdown
                        Object.entries(data.indices).forEach(([name, info]) => {
                            const option = document.createElement('option');
                            option.value = name;
                            option.textContent = `${name} (${info.document_count} docs)`;
                            if (name === data.current_index) {
                                option.selected = true;
                                currentIndexName = name;
                                updateIndexInfo(info);
                            }
                            select.appendChild(option);
                        });
                        
                        // Enable/disable delete button
                        const deleteBtn = document.getElementById('delete-index-btn');
                        deleteBtn.style.display = currentIndexName ? 'inline-block' : 'none';
                    }
                } catch (error) {
                    console.error('Error loading indices:', error);
                }
            }
            
            function updateIndexInfo(info) {
                const infoSpan = document.getElementById('index-info');
                if (info) {
                    infoSpan.textContent = `${info.document_count} documents, ${info.metadata.total_files || 0} files`;
                } else {
                    infoSpan.textContent = '';
                }
            }
            
            async function loadSelectedIndex() {
                const select = document.getElementById('index-select');
                const indexName = select.value;
                
                if (!indexName) return;
                
                const status = document.getElementById('status');
                status.className = 'status loading';
                status.textContent = 'Loading index...';
                status.style.display = 'block';
                
                try {
                    const response = await fetch(`/api/indices/${indexName}/load`, {
                        method: 'POST'
                    });
                    const data = await response.json();
                    
                    if (response.ok) {
                        currentIndexName = indexName;
                        status.className = 'status success';
                        status.textContent = 'Index loaded successfully. Refreshing visualization...';
                        
                        // Update index info
                        const indicesResp = await fetch('/api/indices');
                        const indicesData = await indicesResp.json();
                        if (indicesData.indices[indexName]) {
                            updateIndexInfo(indicesData.indices[indexName]);
                        }
                        
                        // Reload the page to reinitialize visualizer with new index
                        setTimeout(() => {
                            window.location.reload();
                        }, 1000);
                    } else {
                        throw new Error(data.error || 'Failed to load index');
                    }
                } catch (error) {
                    status.className = 'status error';
                    status.textContent = 'Error: ' + error.message;
                    console.error('Error loading index:', error);
                }
            }
            
            async function deleteSelectedIndex() {
                const indexName = currentIndexName;
                if (!indexName) return;
                
                if (!confirm(`Are you sure you want to delete the index "${indexName}"? This action cannot be undone.`)) {
                    return;
                }
                
                const status = document.getElementById('status');
                status.className = 'status loading';
                status.textContent = 'Deleting index...';
                status.style.display = 'block';
                
                try {
                    const response = await fetch(`/api/indices/${indexName}`, {
                        method: 'DELETE'
                    });
                    const data = await response.json();
                    
                    if (response.ok) {
                        status.className = 'status success';
                        status.textContent = 'Index deleted successfully';
                        await loadIndices();
                        setTimeout(() => {
                            status.style.display = 'none';
                        }, 3000);
                    } else {
                        throw new Error(data.error || 'Failed to delete index');
                    }
                } catch (error) {
                    status.className = 'status error';
                    status.textContent = 'Error: ' + error.message;
                }
            }
            
            function refreshIndices() {
                loadIndices();
            }
            
            // Create Index Functions
            function showCreateIndexModal() {
                document.getElementById('create-index-modal').style.display = 'block';
                document.getElementById('new-index-name').value = '';
                document.getElementById('new-index-path').value = '';
                document.getElementById('indexing-progress').style.display = 'none';
                document.getElementById('indexing-error').style.display = 'none';
            }
            
            function closeCreateIndexModal() {
                document.getElementById('create-index-modal').style.display = 'none';
                if (indexingInterval) {
                    clearInterval(indexingInterval);
                    indexingInterval = null;
                }
            }
            
            async function createNewIndex() {
                const indexName = document.getElementById('new-index-name').value.trim();
                const folderPath = document.getElementById('new-index-path').value.trim();
                
                if (!indexName || !folderPath) {
                    alert('Please provide both index name and folder path');
                    return;
                }
                
                const errorDiv = document.getElementById('indexing-error');
                const progressDiv = document.getElementById('indexing-progress');
                
                errorDiv.style.display = 'none';
                progressDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/api/index-folder', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            index_name: indexName,
                            folder_path: folderPath
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        indexingTaskId = data.task_id;
                        // Start polling for progress
                        indexingInterval = setInterval(checkIndexingProgress, 1000);
                    } else {
                        throw new Error(data.error || 'Failed to start indexing');
                    }
                } catch (error) {
                    errorDiv.textContent = error.message;
                    errorDiv.style.display = 'block';
                    progressDiv.style.display = 'none';
                }
            }
            
            async function checkIndexingProgress() {
                if (!indexingTaskId) return;
                
                try {
                    const response = await fetch(`/api/indexing-progress/${indexingTaskId}`);
                    const data = await response.json();
                    
                    if (response.ok) {
                        const progress = data.progress;
                        const progressText = document.getElementById('indexing-progress-text');
                        const progressBar = document.getElementById('indexing-progress-bar');
                        
                        if (progress.total > 0) {
                            const percentage = (progress.progress / progress.total) * 100;
                            progressText.textContent = `${progress.progress}/${progress.total} chunks`;
                            progressBar.style.width = percentage + '%';
                        }
                        
                        if (progress.status === 'completed') {
                            clearInterval(indexingInterval);
                            indexingInterval = null;
                            progressText.textContent = `Completed! ${progress.result.document_count} documents indexed.`;
                            
                            // Reload indices and close modal after delay
                            setTimeout(async () => {
                                await loadIndices();
                                closeCreateIndexModal();
                                
                                // Load the new index
                                const select = document.getElementById('index-select');
                                select.value = document.getElementById('new-index-name').value.trim();
                                await loadSelectedIndex();
                            }, 2000);
                        } else if (progress.status === 'error') {
                            clearInterval(indexingInterval);
                            indexingInterval = null;
                            const errorDiv = document.getElementById('indexing-error');
                            errorDiv.textContent = progress.error || 'Indexing failed';
                            errorDiv.style.display = 'block';
                        }
                    }
                } catch (error) {
                    console.error('Error checking progress:', error);
                }
            }
            
            // Search Functions
            async function searchDocuments() {
                const query = document.getElementById('search-query').value.trim();
                const topK = parseInt(document.getElementById('search-top-k').value) || 10;
                
                if (!query) {
                    alert('Please enter a search query');
                    return;
                }
                
                const status = document.getElementById('status');
                status.className = 'status loading';
                status.textContent = 'Searching...';
                status.style.display = 'block';
                
                try {
                    const response = await fetch('/api/search', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            query: query,
                            top_k: topK
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        displaySearchResults(data.results);
                        status.className = 'status success';
                        status.textContent = `Found ${data.result_count} results`;
                        setTimeout(() => {
                            status.style.display = 'none';
                        }, 3000);
                    } else {
                        throw new Error(data.error || 'Search failed');
                    }
                } catch (error) {
                    status.className = 'status error';
                    status.textContent = 'Error: ' + error.message;
                }
            }
            
            function displaySearchResults(results) {
                const resultsDiv = document.getElementById('search-results');
                const contentDiv = document.getElementById('search-results-content');
                
                if (results.length === 0) {
                    contentDiv.innerHTML = '<p>No results found.</p>';
                } else {
                    contentDiv.innerHTML = results.map((result, i) => `
                        <div style="margin-bottom: 20px; padding: 15px; background-color: white; border-radius: 8px; border-left: 4px solid #20c997;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                <strong>Result ${i + 1}</strong>
                                <span style="color: #666;">Score: ${result.score.toFixed(4)}</span>
                            </div>
                            <div style="color: #0066cc; margin-bottom: 5px;">
                                📄 ${result.file} (Chunk ${result.chunk_index})
                            </div>
                            <div style="color: #333; line-height: 1.6;">
                                ${result.text}
                            </div>
                        </div>
                    `).join('');
                }
                
                resultsDiv.style.display = 'block';
            }
            
            function clearSearchResults() {
                document.getElementById('search-query').value = '';
                document.getElementById('search-results').style.display = 'none';
            }
            
            // Load initial visualization
            window.onload = function() {
                // Load configurations, cluster states, and indices
                Promise.all([
                    loadConfigurations(),
                    loadClusterStates(),
                    loadIndices()
                ]).then(([configData, stateData]) => {
                    // Check if there's a last used state to load
                    if (stateData && stateData.last_used && savedStates[stateData.last_used]) {
                        console.log(`Auto-loading last used state: ${stateData.last_used}`);
                        // Select the last used state in the dropdown
                        document.getElementById('state-select').value = stateData.last_used;
                        // Load the state with error handling
                        loadSelectedState(true).catch(error => {
                            console.error('Failed to load last used state:', error);
                            // Fall back to default visualization
                            updateVisualizationWithParams();
                        });
                    } else {
                        // No last state, check for default config
                        if (!defaultConfig || !savedConfigurations[defaultConfig]) {
                            updateVisualizationWithParams();
                        }
                    }
                }).catch(error => {
                    console.error('Failed to load initial configuration:', error);
                    // Fall back to default visualization
                    updateVisualizationWithParams();
                });
            };
        </script>
    </body>
    </html>
    '''
    
    return render_template_string(html_template)

@app.route('/api/compute', methods=['POST'])
def compute_visualization():
    """Compute visualization with specified parameters"""
    try:
        params = request.json
        print(f"[COMPUTE] Received params: {params}")
        
        algorithm = params.get('algorithm', 'tsne')
        print(f"[COMPUTE] Algorithm: {algorithm}")
        
        # Create cache key from parameters
        cache_key = f"{algorithm}_{json.dumps(params, sort_keys=True)}"
        
        # Compute reduction
        print(f"[COMPUTE] Computing reduction with cache_key: {cache_key}")
        coordinates = visualizer.compute_reduction(algorithm, params, cache_key)
        print(f"[COMPUTE] Coordinates shape: {coordinates.shape}, dtype: {coordinates.dtype}")
        
        # Store the results for plotting
        visualizer.tsne_results = coordinates  # Reuse existing attribute
        
        # Create plot data
        color_by = params.get('color_by', 'file')
        cluster_params = params.get('cluster_params', {})
        print(f"[COMPUTE] Creating plot data with color_by: {color_by}, cluster_params: {cluster_params}")
        plot_data = visualizer.create_plot_data(coordinates, color_by, cluster_params)
        
        # Log the structure of plot_data to find numpy types
        print(f"[COMPUTE] Plot data keys: {plot_data.keys()}")
        print(f"[COMPUTE] Number of traces: {len(plot_data.get('data', []))}")
        
        # Deep check for numpy types in plot_data
        def check_for_numpy_types(obj, path=""):
            if isinstance(obj, (np.integer, np.floating)):
                print(f"[COMPUTE] Found numpy type at {path}: {type(obj).__name__} = {obj}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    check_for_numpy_types(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_for_numpy_types(item, f"{path}[{i}]")
        
        print("[COMPUTE] Checking for numpy types in plot_data...")
        check_for_numpy_types(plot_data)
        
        # Sanitize plot_data before returning
        plot_data_sanitized = sanitize_for_json(plot_data)
        
        response_data = {
            'status': 'success',
            'plot': plot_data_sanitized,
            'algorithm': algorithm,
            'dimensions': [int(d) for d in coordinates.shape]  # Convert numpy int64 to regular int
        }
        
        print("[COMPUTE] Response prepared successfully")
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        print(f"[COMPUTE] Error in compute_visualization: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    """Get list of available algorithms and their parameters"""
    return jsonify({
        'algorithms': {
            'tsne': {
                'name': 't-SNE',
                'parameters': {
                    'perplexity': {'min': 5, 'max': 50, 'default': 30},
                    'learning_rate': {'min': 10, 'max': 1000, 'default': 200},
                    'n_iter': {'min': 250, 'max': 5000, 'default': 1000},
                    'early_exaggeration': {'min': 4, 'max': 50, 'default': 12}
                }
            },
            'umap': {
                'name': 'UMAP',
                'parameters': {
                    'n_neighbors': {'min': 5, 'max': 50, 'default': 15},
                    'min_dist': {'min': 0.0, 'max': 1.0, 'default': 0.1},
                    'metric': {'options': ['euclidean', 'cosine', 'manhattan'], 'default': 'euclidean'}
                }
            },
            'pca': {
                'name': 'PCA',
                'parameters': {}
            }
        }
    })

@app.route('/api/summarize-clusters', methods=['POST'])
def summarize_clusters():
    """Generate summaries for current clusters"""
    try:
        print("Summarize clusters endpoint called")
        if not GEMINI_API_KEY:
            print("No GEMINI_API_KEY found")
            return jsonify({
                'status': 'error',
                'error': 'GEMINI_API_KEY not configured. Please set the environment variable.'
            }), 500
        
        params = request.json
        print(f"Received params: {params}")
        algorithm = params.get('algorithm')
        color_by = params.get('color_by')
        cluster_context = params.get('cluster_context', {})
        cluster_params = params.get('cluster_params', {})
        
        # Check if clustering is active
        if color_by not in ['kmeans', 'dbscan', 'hierarchical']:
            return jsonify({
                'status': 'error',
                'error': 'Clustering must be active to generate summaries'
            }), 400
        
        # Check if we're in a subcluster view
        current_path = cluster_context.get('current_path', 'root')
        parent_cluster_id = cluster_context.get('parent_cluster_id')
        original_indices = cluster_context.get('original_indices')
        subset_indices = None
        
        if current_path != 'root':
            print(f"Summarizing subcluster - path: {current_path}")
            
            # Check if we have original indices from the current cluster
            if original_indices is not None:
                # Use the provided original indices for the current drill-down view
                subset_indices = np.array(original_indices)
                print(f"Using provided original indices: {len(subset_indices)} documents in current drill-down cluster")
            elif parent_cluster_id is not None:
                # Fallback to computing from parent cluster
                print(f"Falling back to parent cluster computation - parent: {parent_cluster_id}")
                
                # We need to get the indices of documents in the parent cluster
                # First, get the parent level clustering
                parent_params = cluster_context.get('parent_params', params)
                parent_algorithm = parent_params.get('algorithm', algorithm)
                parent_color_by = parent_params.get('color_by', color_by)
                parent_cluster_params = parent_params.get('cluster_params', {})
                
                # Compute parent coordinates and labels
                parent_cache_key = f"{parent_algorithm}_{json.dumps(parent_params, sort_keys=True)}"
                parent_coordinates = visualizer.compute_reduction(parent_algorithm, parent_params, parent_cache_key)
                parent_labels = visualizer.apply_clustering(parent_coordinates, parent_color_by, parent_cluster_params)
                
                # Get indices of documents in the parent cluster
                subset_indices = np.where(parent_labels == parent_cluster_id)[0]
                print(f"Found {len(subset_indices)} documents in parent cluster {parent_cluster_id}")
            
            # Now we need to work with only these documents
            if subset_indices is not None:
                # Temporarily replace visualizer data
                original_documents = visualizer.documents
                original_embeddings = visualizer.embeddings
                
                try:
                    subset_documents = [visualizer.documents[i] for i in subset_indices]
                    subset_embeddings = visualizer.embeddings[subset_indices]
                    
                    visualizer.documents = subset_documents
                    visualizer.embeddings = subset_embeddings
                    
                    # Create cache key for the subset
                    cache_key = f"{algorithm}_{json.dumps(params, sort_keys=True)}"
                    
                    # Recompute coordinates and labels for the subset
                    coordinates = visualizer.compute_reduction(algorithm, params, cache_key + f"_cluster_{current_path}")
                    labels = visualizer.apply_clustering(coordinates, color_by, cluster_params)
                    
                    # Generate summaries for the subset
                    summaries = visualizer.summarize_clusters(coordinates, labels)
                    
                finally:
                    # Restore original data
                    visualizer.documents = original_documents
                    visualizer.embeddings = original_embeddings
            else:
                # No subset indices available, return empty summaries
                print("Warning: No subset indices available for drill-down summarization")
                summaries = {}
        else:
            # Root level - summarize all clusters normally
            print("Summarizing root level clusters")
            cache_key = f"{algorithm}_{json.dumps(params, sort_keys=True)}"
            coordinates = visualizer.compute_reduction(algorithm, params, cache_key)
            
            # Apply clustering to get labels
            print(f"Applying clustering: {color_by} with params: {cluster_params}")
            labels = visualizer.apply_clustering(coordinates, color_by, cluster_params)
            print(f"Got labels: unique values = {np.unique(labels)}")
            
            # Generate summaries
            summaries = visualizer.summarize_clusters(coordinates, labels)
        
        print(f"Generated {len(summaries)} summaries")
        
        return jsonify({
            'status': 'success',
            'summaries': summaries,
            'total_clusters': len(summaries),
            'cluster_path': current_path
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/configurations', methods=['GET'])
def get_configurations():
    """Get all saved configurations"""
    configs = load_configurations()
    return jsonify({
        'status': 'success',
        'configurations': configs['configurations'],
        'default': configs['default']
    })

@app.route('/api/configurations', methods=['POST'])
def save_configuration():
    """Save a new configuration"""
    try:
        data = request.json
        name = data.get('name')
        params = data.get('params')
        set_as_default = data.get('setAsDefault', False)
        
        if not name or not params:
            return jsonify({
                'status': 'error',
                'error': 'Name and parameters are required'
            }), 400
        
        configs = load_configurations()
        
        # Sanitize params to handle numpy types
        sanitized_params = sanitize_for_json(params)
        
        # Add timestamp to configuration
        sanitized_params['saved_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save configuration
        configs['configurations'][name] = sanitized_params
        
        # Set as default if requested
        if set_as_default:
            configs['default'] = name
        
        if save_configurations(configs):
            return jsonify({
                'status': 'success',
                'message': f'Configuration "{name}" saved successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Failed to save configuration'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/configurations/<name>', methods=['DELETE'])
def delete_configuration(name):
    """Delete a configuration"""
    try:
        configs = load_configurations()
        
        if name not in configs['configurations']:
            return jsonify({
                'status': 'error',
                'error': f'Configuration "{name}" not found'
            }), 404
        
        # Remove configuration
        del configs['configurations'][name]
        
        # Clear default if it was this configuration
        if configs['default'] == name:
            configs['default'] = None
        
        if save_configurations(configs):
            return jsonify({
                'status': 'success',
                'message': f'Configuration "{name}" deleted successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Failed to delete configuration'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/configurations/default', methods=['PUT'])
def set_default_configuration():
    """Set the default configuration"""
    try:
        data = request.json
        name = data.get('name')
        
        configs = load_configurations()
        
        if name and name not in configs['configurations']:
            return jsonify({
                'status': 'error',
                'error': f'Configuration "{name}" not found'
            }), 404
        
        configs['default'] = name
        
        if save_configurations(configs):
            return jsonify({
                'status': 'success',
                'message': f'Default configuration set to "{name}"' if name else 'Default configuration cleared'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Failed to set default configuration'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/analyze-cluster', methods=['POST'])
def analyze_cluster():
    """Analyze a specific cluster with inherited parent parameters"""
    try:
        params = request.json
        current_params = params.get('current_params')
        cluster_id = params.get('cluster_id')
        
        # Get current visualization to find cluster documents
        current_algorithm = current_params.get('algorithm', 'umap')
        current_color_by = current_params.get('color_by', 'kmeans')
        
        # Compute current state to get cluster labels
        cache_key = f"{current_algorithm}_{json.dumps(current_params, sort_keys=True)}"
        coordinates = visualizer.compute_reduction(current_algorithm, current_params, cache_key)
        
        # Apply clustering to get labels
        current_cluster_params = current_params.get('cluster_params', {})
        labels = visualizer.apply_clustering(coordinates, current_color_by, current_cluster_params)
        
        # Find documents in the selected cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            return jsonify({
                'status': 'error',
                'error': f'No documents found in cluster {cluster_id}'
            }), 404
        
        # Get subset of documents and embeddings
        subset_documents = [visualizer.documents[i] for i in cluster_indices]
        subset_embeddings = visualizer.embeddings[cluster_indices]
        
        # Temporarily replace visualizer data
        original_documents = visualizer.documents
        original_embeddings = visualizer.embeddings
        
        try:
            visualizer.documents = subset_documents
            visualizer.embeddings = subset_embeddings
            
            # IMPORTANT: Use the SAME algorithm and parameters as parent
            # Extract only the algorithm-specific params from current_params
            parent_algorithm = current_params.get('algorithm', 'umap')
            parent_n_components = current_params.get('n_components', 2)
            
            # Build parameters matching parent configuration exactly
            subcluster_params = {
                'algorithm': parent_algorithm,
                'n_components': parent_n_components
            }
            
            # Copy algorithm-specific parameters from parent
            if parent_algorithm == 'umap':
                for key in ['n_neighbors', 'min_dist', 'metric']:
                    if key in current_params:
                        subcluster_params[key] = current_params[key]
            elif parent_algorithm == 'tsne':
                for key in ['perplexity', 'learning_rate', 'n_iter', 'early_exaggeration']:
                    if key in current_params:
                        subcluster_params[key] = current_params[key]
            
            # Apply parent's dimensionality reduction to subset
            new_coordinates = visualizer.compute_reduction(
                parent_algorithm, 
                subcluster_params,
                cache_key=f"cluster_{cluster_id}_{parent_algorithm}_inherited"
            )
            
            # Apply parent's clustering algorithm to subset
            parent_color_by = current_params.get('color_by', 'kmeans')
            parent_cluster_params = current_params.get('cluster_params', {})
            new_labels = visualizer.apply_clustering(new_coordinates, parent_color_by, parent_cluster_params)
            
            # Create plot data using parent's configuration
            plot_data = visualizer.create_plot_data(new_coordinates, parent_color_by, parent_cluster_params)
            
            # Update plot title
            plot_data['layout']['title'] = f'Cluster {cluster_id} Analysis ({len(subset_documents)} documents)'
            
            # Generate breadcrumbs
            current_path = params.get('current_path', 'root')
            breadcrumbs = [{'id': 'root', 'label': 'All Documents'}]
            
            if current_path != 'root':
                path_parts = current_path.split('-')
                for i, part in enumerate(path_parts):
                    breadcrumbs.append({
                        'id': '-'.join(path_parts[:i+1]),
                        'label': f'Cluster {part}'
                    })
            
            breadcrumbs.append({
                'id': f"{current_path}-{cluster_id}" if current_path != 'root' else str(cluster_id),
                'label': f'Cluster {cluster_id}'
            })
            
            return jsonify({
                'status': 'success',
                'plot': plot_data,
                'breadcrumbs': breadcrumbs,
                'cluster_info': {
                    'id': cluster_id,
                    'doc_count': len(subset_documents),
                    'original_indices': cluster_indices.tolist()
                },
                'visualization_params': subcluster_params,
                'parent_params': current_params,
                'new_path': f"{current_path}-{cluster_id}" if current_path != 'root' else str(cluster_id)
            })
            
        finally:
            # Restore original data
            visualizer.documents = original_documents
            visualizer.embeddings = original_embeddings
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/recursive-cluster', methods=['POST'])
def recursive_cluster():
    """Compute recursive clustering hierarchy"""
    try:
        params = request.json
        algorithm = params.get('algorithm', 'umap')
        cluster_algorithm = params.get('cluster_algorithm', 'kmeans')
        max_depth = params.get('max_depth', 3)
        cluster_params = params.get('cluster_params', {})
        dim_reduction_params = params.get('dim_reduction_params', {})
        
        # Store current params for reuse
        session_id = str(uuid.uuid4())
        computation_status[session_id] = {
            'status': 'computing',
            'progress': 0,
            'hierarchy': None
        }
        
        # Compute recursive clustering
        hierarchy = visualizer.apply_recursive_clustering(
            max_depth=max_depth,
            algorithm=cluster_algorithm,
            dim_reduction_algorithm=algorithm,
            cluster_params=cluster_params,
            dim_reduction_params=dim_reduction_params
        )
        
        # Store hierarchy for later use
        computation_status[session_id] = {
            'status': 'complete',
            'progress': 100,
            'hierarchy': hierarchy
        }
        
        # Also store in cache for quick access
        cache.set(f'hierarchy_{session_id}', hierarchy, timeout=3600)
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'hierarchy': hierarchy,
            'total_nodes': count_nodes(hierarchy)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/cluster-drill-down', methods=['POST'])
def cluster_drill_down():
    """Get visualization data for a specific cluster in the hierarchy"""
    try:
        params = request.json
        session_id = params.get('session_id')
        cluster_path = params.get('cluster_path', 'root')
        
        # Get hierarchy from cache
        hierarchy = cache.get(f'hierarchy_{session_id}')
        if not hierarchy:
            return jsonify({
                'status': 'error',
                'error': 'Session not found or expired'
            }), 404
        
        # Navigate to the requested cluster
        node = find_cluster_node(hierarchy, cluster_path)
        if not node:
            return jsonify({
                'status': 'error',
                'error': f'Cluster {cluster_path} not found'
            }), 404
        
        # Create plot data for this cluster
        plot_data = create_drill_down_plot(node)
        
        # Get breadcrumb path
        breadcrumbs = get_breadcrumb_path(cluster_path)
        
        return jsonify({
            'status': 'success',
            'plot': plot_data,
            'breadcrumbs': breadcrumbs,
            'cluster_info': {
                'id': node['cluster_id'],
                'level': node['level'],
                'doc_count': node['doc_count'],
                'has_children': len(node['children']) > 0
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

def count_nodes(hierarchy):
    """Count total nodes in hierarchy"""
    count = 1  # Count current node
    for child in hierarchy.get('children', []):
        count += count_nodes(child)
    return count

def find_cluster_node(hierarchy, path):
    """Find a cluster node by path (e.g., 'root', '0', '0-1', '0-1-2')"""
    if path == 'root':
        return hierarchy
    
    path_parts = path.split('-')
    current = hierarchy
    
    for part in path_parts:
        found = False
        for child in current.get('children', []):
            if child['cluster_id'].endswith(f'-{part}') or child['cluster_id'] == part:
                current = child
                found = True
                break
        if not found:
            return None
    
    return current

def get_breadcrumb_path(cluster_path):
    """Generate breadcrumb navigation from cluster path"""
    if cluster_path == 'root':
        return [{'id': 'root', 'label': 'All Documents'}]
    
    breadcrumbs = [{'id': 'root', 'label': 'All Documents'}]
    path_parts = cluster_path.split('-')
    
    for i, part in enumerate(path_parts):
        cluster_id = '-'.join(path_parts[:i+1])
        breadcrumbs.append({
            'id': cluster_id,
            'label': f'Cluster {part}'
        })
    
    return breadcrumbs

def create_drill_down_plot(node):
    """Create plot data for a specific cluster node"""
    if 'coordinates' not in node or not node['coordinates']:
        return {'data': [], 'layout': {}}
    
    coordinates = np.array(node['coordinates'])
    doc_indices = node['doc_indices']
    
    # Create traces for each child cluster
    traces = []
    
    if node['children']:
        # Group documents by child cluster
        child_map = {}
        for child in node['children']:
            child_map[child['cluster_id']] = child['doc_indices']
        
        # Create trace for each child
        for i, child in enumerate(node['children']):
            child_indices = child['doc_indices']
            # Find which points in coordinates correspond to these documents
            coord_indices = [doc_indices.index(idx) for idx in child_indices if idx in doc_indices]
            
            if coord_indices:
                # Get document info for hover text
                hover_texts = []
                for idx in child_indices:
                    if idx < len(visualizer.documents):
                        doc = visualizer.documents[idx]
                        hover_texts.append(
                            f"File: {doc['relative_path']}<br>"
                            f"Chunk: {int(doc['chunk_index'])}<br>"
                            f"Preview: {doc['text'][:100]}...<br>"
                            f"Click to drill down"
                        )
                    else:
                        hover_texts.append(f"Doc {idx}<br>Click to drill down")
                
                trace = {
                    'x': coordinates[coord_indices, 0].tolist(),
                    'y': coordinates[coord_indices, 1].tolist(),
                    'mode': 'markers',
                    'type': 'scatter',
                    'name': f"Cluster {child['cluster_id'].split('-')[-1]}",
                    'marker': {
                        'size': 8,
                        'opacity': 0.7,
                        'color': f'rgb({(i * 67) % 255}, {(i * 97) % 255}, {(i * 137) % 255})'
                    },
                    'text': hover_texts[:len(coord_indices)],
                    'customdata': child['cluster_id'],
                    'hovertemplate': '%{text}<extra></extra>'
                }
                traces.append(trace)
    else:
        # Leaf node - show all documents
        hover_texts = []
        for idx in doc_indices:
            if idx < len(visualizer.documents):
                doc = visualizer.documents[idx]
                hover_texts.append(
                    f"File: {doc['relative_path']}<br>"
                    f"Chunk: {int(doc['chunk_index'])}<br>"
                    f"Preview: {doc['text'][:100]}..."
                )
            else:
                hover_texts.append(f"Doc {idx}")
        
        trace = {
            'x': coordinates[:, 0].tolist(),
            'y': coordinates[:, 1].tolist(),
            'mode': 'markers',
            'type': 'scatter',
            'name': 'Documents',
            'marker': {
                'size': 8,
                'opacity': 0.7,
                'color': 'blue'
            },
            'text': hover_texts,
            'hovertemplate': '%{text}<extra></extra>'
        }
        traces.append(trace)
    
    layout = {
        'title': f'Cluster {node["cluster_id"]} ({node["doc_count"]} documents)',
        'xaxis': {'title': 'Component 1'},
        'yaxis': {'title': 'Component 2'},
        'hovermode': 'closest',
        'height': 600
    }
    
    return {'data': traces, 'layout': layout}

@app.route('/api/delete-clusters', methods=['POST'])
def delete_clusters():
    """Delete selected clusters from the document index"""
    try:
        params = request.json
        clusters_to_delete = params.get('clusters', [])
        algorithm = params.get('algorithm')
        color_by = params.get('color_by')
        
        if not clusters_to_delete:
            return jsonify({
                'status': 'error',
                'error': 'No clusters specified for deletion'
            }), 400
        
        # Get current coordinates and labels
        cache_key = f"{algorithm}_{json.dumps(params, sort_keys=True)}"
        coordinates = visualizer.compute_reduction(algorithm, params, cache_key)
        
        # Apply clustering to get labels
        cluster_params = params.get('cluster_params', {})
        labels = visualizer.apply_clustering(coordinates, color_by, cluster_params)
        
        # Find indices of documents to keep
        indices_to_keep = []
        for i, label in enumerate(labels):
            if label not in clusters_to_delete:
                indices_to_keep.append(i)
        
        # Update the visualizer's documents and embeddings
        new_documents = [visualizer.documents[i] for i in indices_to_keep]
        new_embeddings = visualizer.embeddings[indices_to_keep]
        
        # Calculate stats before updating
        original_count = len(visualizer.documents)
        documents_removed = original_count - len(new_documents)
        
        # Update the visualizer
        visualizer.documents = new_documents
        visualizer.embeddings = new_embeddings
        
        # Save updated index
        updated_index = {
            "documents": new_documents,
            "metadata": {
                "embedding_dim": visualizer.embeddings.shape[1],
                "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_modified": time.strftime("%Y-%m-%d %H:%M:%S"),
                "clusters_deleted": clusters_to_delete
            }
        }
        
        with open("document_index.json", 'w') as f:
            json.dump(updated_index, f)
        
        # Clear cache to force recomputation
        cache.clear()
        
        print(f"Deleted clusters: {clusters_to_delete}")
        print(f"Documents remaining: {len(new_documents)}")
        
        return jsonify({
            'status': 'success',
            'deleted_clusters': clusters_to_delete,
            'documents_removed': documents_removed,
            'documents_remaining': len(new_documents)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/cluster-states', methods=['GET'])
def get_cluster_states():
    """Get all saved cluster states"""
    states = load_cluster_states()
    return jsonify({
        'status': 'success',
        'states': states['states'],
        'last_used': states.get('last_used')
    })

@app.route('/api/cluster-states', methods=['POST'])
def save_cluster_state():
    """Save a complete cluster state"""
    try:
        data = request.json
        name = data.get('name')
        description = data.get('description', '')
        
        if not name:
            return jsonify({
                'status': 'error',
                'error': 'Name is required'
            }), 400
        
        # Get the current state data from the frontend
        state_data = {
            'name': name,
            'description': description,
            'saved_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'visualization_params': data.get('visualization_params', {}),
            'hierarchical_summaries': data.get('hierarchical_summaries', {}),
            'cluster_history': data.get('cluster_history', []),
            'current_path': data.get('current_path', 'root'),
            'current_data': data.get('current_data'),
            'navigation_tree': data.get('navigation_tree', {}),  # Save the navigation tree
            'document_index_hash': document_index_hash,  # Save current document index hash
            'document_count': len(visualizer.documents)  # Save document count for validation
        }
        
        states = load_cluster_states()
        states['states'][name] = state_data
        states['last_used'] = name  # Track last used state
        
        if save_cluster_states(states):
            return jsonify({
                'status': 'success',
                'message': f'Cluster state "{name}" saved successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Failed to save cluster state'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/cluster-states/<name>', methods=['GET'])
def get_cluster_state(name):
    """Load a specific cluster state with validation"""
    states = load_cluster_states()
    
    if name not in states['states']:
        return jsonify({
            'status': 'error',
            'error': f'Cluster state "{name}" not found'
        }), 404
    
    # Validate state against current document index
    max_doc_index = len(visualizer.documents) - 1
    state = states['states'][name]
    is_valid, cleaned_state, validation_errors = validate_cluster_state(state, max_doc_index)
    
    # Check if document index has changed
    saved_hash = state.get('document_index_hash')
    saved_count = state.get('document_count', 0)
    
    if saved_hash != document_index_hash:
        validation_errors.append(
            f"Document index has changed since this state was saved. "
            f"Saved with {saved_count} documents, now have {len(visualizer.documents)}."
        )
    
    # Add validation warnings to response
    response_data = {
        'status': 'success',
        'state': cleaned_state,
        'document_index_changed': saved_hash != document_index_hash
    }
    
    if not is_valid or saved_hash != document_index_hash:
        response_data['validation_warnings'] = validation_errors
        print(f"State '{name}' had validation issues: {validation_errors}")
        
        # Update saved state with cleaned version
        states['states'][name] = cleaned_state
        save_cluster_states(states)
    
    # Update last used
    states['last_used'] = name
    save_cluster_states(states)
    
    return jsonify(response_data)

@app.route('/api/cluster-states/<name>', methods=['DELETE'])
def delete_cluster_state(name):
    """Delete a saved cluster state"""
    try:
        states = load_cluster_states()
        
        if name not in states['states']:
            return jsonify({
                'status': 'error',
                'error': f'Cluster state "{name}" not found'
            }), 404
        
        del states['states'][name]
        
        if save_cluster_states(states):
            return jsonify({
                'status': 'success',
                'message': f'Cluster state "{name}" deleted successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Failed to delete cluster state'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/reset-states', methods=['POST'])
def reset_all_states():
    """Reset all saved states and configurations"""
    try:
        # Reset cluster states
        empty_states = {'states': {}, 'last_used': None}
        if not save_cluster_states(empty_states):
            return jsonify({
                'status': 'error',
                'error': 'Failed to reset cluster states'
            }), 500
        
        # Optionally reset configurations too
        reset_configs = request.json.get('reset_configurations', False) if request.json else False
        if reset_configs:
            empty_configs = {'configurations': {}, 'default': None}
            if not save_configurations(empty_configs):
                return jsonify({
                    'status': 'error',
                    'error': 'Failed to reset configurations'
                }), 500
        
        # Clear server-side cache
        cache.clear()
        
        return jsonify({
            'status': 'success',
            'message': 'All states have been reset successfully',
            'reset_configurations': reset_configs
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/indices', methods=['GET'])
def list_indices():
    """List all available document indices"""
    try:
        registry = load_index_registry()
        return jsonify({
            'status': 'success',
            'indices': registry.get('indices', {}),
            'default_index': registry.get('default_index'),
            'current_index': current_index_name
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/indices/<name>', methods=['GET'])
def get_index_info(name):
    """Get detailed information about a specific index"""
    try:
        registry = load_index_registry()
        if name not in registry.get('indices', {}):
            return jsonify({
                'status': 'error',
                'error': f'Index "{name}" not found'
            }), 404
        
        index_info = registry['indices'][name]
        return jsonify({
            'status': 'success',
            'index': index_info
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/indices/<name>/load', methods=['POST'])
def load_index(name):
    """Load a specific index for visualization"""
    try:
        registry = load_index_registry()
        if name not in registry.get('indices', {}):
            return jsonify({
                'status': 'error',
                'error': f'Index "{name}" not found'
            }), 404
        
        index_info = registry['indices'][name]
        index_file = index_info['file_path']
        
        # Re-initialize visualizer with the new index
        global current_index_name
        initialize_visualizer(index_file)
        current_index_name = name
        
        # Clear cache to force recomputation with new data
        cache.clear()
        
        return jsonify({
            'status': 'success',
            'message': f'Successfully loaded index "{name}"',
            'document_count': len(visualizer.documents) if visualizer else 0
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/indices/<name>', methods=['DELETE'])
def delete_index(name):
    """Delete a document index"""
    try:
        registry = load_index_registry()
        
        if name not in registry.get('indices', {}):
            return jsonify({
                'status': 'error',
                'error': f'Index "{name}" not found'
            }), 404
        
        # Don't allow deleting the current index
        if name == current_index_name:
            return jsonify({
                'status': 'error',
                'error': 'Cannot delete the currently loaded index'
            }), 400
        
        # Delete the index file
        index_info = registry['indices'][name]
        index_file = index_info['file_path']
        if os.path.exists(index_file):
            os.remove(index_file)
        
        # Remove from registry
        del registry['indices'][name]
        
        # Update default if needed
        if registry.get('default_index') == name:
            registry['default_index'] = None
        
        save_index_registry(registry)
        
        return jsonify({
            'status': 'success',
            'message': f'Successfully deleted index "{name}"'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/index-folder', methods=['POST'])
def index_folder():
    """Create a new index from a folder"""
    try:
        data = request.json
        folder_path = data.get('folder_path')
        index_name = data.get('index_name')
        
        if not folder_path or not index_name:
            return jsonify({
                'status': 'error',
                'error': 'folder_path and index_name are required'
            }), 400
        
        # Validate folder path
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            return jsonify({
                'status': 'error',
                'error': f'Invalid folder path: {folder_path}'
            }), 400
        
        # Check if index name already exists
        registry = load_index_registry()
        if index_name in registry.get('indices', {}):
            return jsonify({
                'status': 'error',
                'error': f'Index "{index_name}" already exists'
            }), 400
        
        # Initialize embedder if needed
        global embedder
        if embedder is None:
            if not GEMINI_API_KEY:
                return jsonify({
                    'status': 'error',
                    'error': 'GEMINI_API_KEY not configured'
                }), 500
            embedder = DocumentEmbedder(GEMINI_API_KEY)
            
        # Start indexing in background
        task_id = str(uuid.uuid4())
        indexing_progress[task_id] = {
            'status': 'starting',
            'progress': 0,
            'total': 0,
            'index_name': index_name,
            'folder_path': folder_path
        }
        
        # Run indexing in a separate thread
        thread = threading.Thread(
            target=_run_indexing,
            args=(task_id, folder_path, index_name)
        )
        thread.start()
        
        return jsonify({
            'status': 'success',
            'task_id': task_id,
            'message': f'Started indexing folder: {folder_path}'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

def _run_indexing(task_id, folder_path, index_name):
    """Run indexing in background thread"""
    try:
        def progress_callback(current, total):
            indexing_progress[task_id]['progress'] = current
            indexing_progress[task_id]['total'] = total
            indexing_progress[task_id]['status'] = 'processing'
        
        # Create the index
        index_data = embedder.index_folder(folder_path, progress_callback)
        
        # Save the index
        index_file_path = os.path.join(INDICES_DIR, f"{index_name}_index.json")
        with open(index_file_path, 'w') as f:
            json.dump(index_data, f)
        
        # Update registry
        registry = load_index_registry()
        registry['indices'][index_name] = {
            'name': index_name,
            'path': folder_path,
            'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'document_count': len(index_data['documents']),
            'chunk_count': len(index_data['documents']),
            'file_path': index_file_path,
            'metadata': index_data.get('metadata', {})
        }
        
        # Set as default if it's the first index
        if not registry.get('default_index'):
            registry['default_index'] = index_name
        
        save_index_registry(registry)
        
        indexing_progress[task_id]['status'] = 'completed'
        indexing_progress[task_id]['result'] = {
            'document_count': len(index_data['documents']),
            'file_count': index_data['metadata'].get('total_files', 0)
        }
        
    except Exception as e:
        indexing_progress[task_id]['status'] = 'error'
        indexing_progress[task_id]['error'] = str(e)
        print(f"Error in indexing task {task_id}: {e}")

@app.route('/api/indexing-progress/<task_id>', methods=['GET'])
def get_indexing_progress(task_id):
    """Get progress of an indexing operation"""
    if task_id not in indexing_progress:
        return jsonify({
            'status': 'error',
            'error': 'Task not found'
        }), 404
    
    return jsonify({
        'status': 'success',
        'progress': indexing_progress[task_id]
    })

@app.route('/api/search', methods=['POST'])
def search_documents():
    """Search within the current loaded index"""
    try:
        if not visualizer or not visualizer.index_data:
            return jsonify({
                'status': 'error',
                'error': 'No index loaded'
            }), 400
        
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', 10)
        
        if not query:
            return jsonify({
                'status': 'error',
                'error': 'Query is required'
            }), 400
        
        if not embedder:
            return jsonify({
                'status': 'error',
                'error': 'Embedder not initialized'
            }), 500
        
        # Perform search
        results = embedder.search(query, visualizer.index_data, top_k)
        
        # Format results
        formatted_results = []
        for score, doc in results:
            formatted_results.append({
                'score': float(score),
                'file': doc.get('relative_path', doc.get('file')),
                'chunk_index': doc['chunk_index'],
                'text': doc['text'][:500] + '...' if len(doc['text']) > 500 else doc['text']
            })
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': formatted_results,
            'result_count': len(formatted_results)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive Document Embeddings Visualization Server')
    parser.add_argument('--reset-states', action='store_true', 
                        help='Reset all saved cluster states before starting')
    parser.add_argument('--reset-all', action='store_true',
                        help='Reset all saved states and configurations before starting')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on (default: 5000)')
    
    args = parser.parse_args()
    
    # Handle reset options
    if args.reset_all or args.reset_states:
        print("Resetting saved states...")
        empty_states = {'states': {}, 'last_used': None}
        save_cluster_states(empty_states)
        
        if args.reset_all:
            print("Resetting configurations...")
            empty_configs = {'configurations': {}, 'default': None}
            save_configurations(empty_configs)
    
    print("Initializing visualizer...")
    initialize_visualizer()
    print(f"Loaded {len(visualizer.documents)} documents")
    print(f"Document index hash: {document_index_hash}")
    print(f"\nStarting server at http://localhost:{args.port}")
    print("Open this URL in your browser to use the interactive visualization")
    
    app.run(debug=True, port=args.port)