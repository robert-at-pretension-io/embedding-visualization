#!/usr/bin/env python3
"""
Interactive t-SNE visualization for document embeddings
"""

import json
import os
import sys
import time
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse


class EmbeddingVisualizer:
    def __init__(self, index_file: str = "document_index.json"):
        self.index_file = index_file
        self.index_data = None
        self.embeddings = None
        self.documents = None
        self.tsne_results = None
        
    def load_index(self) -> bool:
        """Load the document index from JSON file."""
        try:
            print(f"Loading index from {self.index_file}...")
            with open(self.index_file, 'r') as f:
                self.index_data = json.load(f)
            
            # Extract embeddings and document info
            self.documents = self.index_data['documents']
            self.embeddings = np.array([doc['embedding'] for doc in self.documents])
            
            print(f"Loaded {len(self.documents)} document chunks")
            print(f"Embedding dimensions: {self.embeddings.shape}")
            return True
            
        except FileNotFoundError:
            print(f"Error: Index file '{self.index_file}' not found.")
            print("Please run document_embedder.py first to create an index.")
            return False
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def compute_tsne(self, n_components: int = 2, perplexity: int = 30, 
                     n_iter: int = 1000, random_state: int = 42,
                     cache_file: Optional[str] = "tsne_cache.npy") -> np.ndarray:
        """
        Compute t-SNE reduction of embeddings.
        
        Args:
            n_components: Number of dimensions (2 or 3)
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations
            random_state: Random seed for reproducibility
            cache_file: Optional file to cache results
        """
        # Check for cached results
        if cache_file and os.path.exists(cache_file):
            try:
                print(f"Loading cached t-SNE results from {cache_file}...")
                self.tsne_results = np.load(cache_file)
                if self.tsne_results.shape[0] == len(self.embeddings):
                    return self.tsne_results
                else:
                    print("Cache size mismatch, recomputing...")
            except:
                print("Failed to load cache, recomputing...")
        
        print(f"Computing t-SNE with perplexity={perplexity}...")
        print("This may take a few minutes for large datasets...")
        
        # Standardize features
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings)
        
        # Apply t-SNE
        tsne = TSNE(n_components=n_components, 
                    perplexity=perplexity,
                    max_iter=n_iter,
                    random_state=random_state,
                    verbose=1)
        
        start_time = time.time()
        self.tsne_results = tsne.fit_transform(embeddings_scaled)
        elapsed_time = time.time() - start_time
        
        print(f"t-SNE completed in {elapsed_time:.2f} seconds")
        
        # Cache results
        if cache_file:
            np.save(cache_file, self.tsne_results)
            print(f"Cached results to {cache_file}")
        
        return self.tsne_results
    
    def create_interactive_plot(self, color_by: str = 'file', 
                               show_convex_hull: bool = False,
                               point_size: int = 8,
                               opacity: float = 0.7) -> go.Figure:
        """
        Create an interactive Plotly scatter plot.
        
        Args:
            color_by: How to color points ('file', 'chunk_index', or None)
            show_convex_hull: Whether to show convex hulls around file groups
            point_size: Size of scatter points
            opacity: Opacity of points
        """
        if self.tsne_results is None:
            raise ValueError("Must compute t-SNE before plotting")
        
        # Prepare data for plotting
        df_data = {
            'x': self.tsne_results[:, 0].tolist(),  # Convert to list for proper JSON serialization
            'y': self.tsne_results[:, 1].tolist(),  # Convert to list for proper JSON serialization
            'file': [doc['relative_path'] for doc in self.documents],
            'chunk_index': [doc['chunk_index'] for doc in self.documents],
            'text_preview': [doc['text'][:200] + '...' if len(doc['text']) > 200 
                           else doc['text'] for doc in self.documents],
            'full_text': [doc['text'] for doc in self.documents]
        }
        
        # Add 3D coordinates if available
        if self.tsne_results.shape[1] == 3:
            df_data['z'] = self.tsne_results[:, 2].tolist()  # Convert to list for proper JSON serialization
        
        df = pd.DataFrame(df_data)
        
        # Create color mapping
        if color_by == 'file':
            unique_files = df['file'].unique()
            colors = px.colors.qualitative.Plotly * (len(unique_files) // len(px.colors.qualitative.Plotly) + 1)
            color_map = {file: colors[i] for i, file in enumerate(unique_files)}
            df['color'] = df['file'].map(color_map)
            df['color_label'] = df['file']
        elif color_by == 'chunk_index':
            df['color'] = df['chunk_index']
            df['color_label'] = df['chunk_index'].astype(str)
        else:
            df['color'] = 'blue'
            df['color_label'] = 'Document'
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot
        if self.tsne_results.shape[1] == 2:
            # 2D plot - simplified version without customdata
            if color_by == 'file':
                # Use plotly express colors
                colors = px.colors.qualitative.Plotly
                unique_files = df['file'].unique()
                
                for i, file in enumerate(unique_files):
                    mask = df['file'] == file
                    color = colors[i % len(colors)]
                    
                    fig.add_trace(go.Scatter(
                        x=df[mask]['x'].tolist(),
                        y=df[mask]['y'].tolist(),
                        mode='markers',
                        name=file if len(file) < 30 else file[:27] + '...',
                        marker=dict(
                            size=point_size,
                            color=color,
                            opacity=opacity,
                        ),
                        text=df[mask].apply(lambda row: f"File: {row['file']}<br>"
                                                       f"Chunk: {row['chunk_index']}<br>"
                                                       f"Preview: {row['text_preview']}", axis=1).tolist(),
                        hovertemplate='%{text}<extra></extra>',
                    ))
            else:
                # Single color for all points
                fig.add_trace(go.Scatter(
                    x=df['x'].tolist(),
                    y=df['y'].tolist(),
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color='blue',
                        opacity=opacity,
                    ),
                    text=df.apply(lambda row: f"File: {row['file']}<br>"
                                             f"Chunk: {row['chunk_index']}<br>"
                                             f"Preview: {row['text_preview']}", axis=1).tolist(),
                    hovertemplate='%{text}<extra></extra>',
                ))
        else:
            # 3D plot
            if color_by:
                for label in df['color_label'].unique():
                    mask = df['color_label'] == label
                    fig.add_trace(go.Scatter3d(
                        x=df[mask]['x'],
                        y=df[mask]['y'],
                        z=df[mask]['z'],
                        mode='markers',
                        name=str(label)[:30] + '...' if len(str(label)) > 30 else str(label),
                        marker=dict(
                            size=point_size,
                            opacity=opacity,
                        ),
                        text=[f"File: {row['file']}<br>"
                              f"Chunk: {row['chunk_index']}<br>"
                              f"Preview: {row['text_preview']}"
                              for _, row in df[mask].iterrows()],
                        hovertemplate='%{text}<extra></extra>',
                        customdata=df[mask]['full_text'],
                    ))
        
        # Update layout
        title = f"Document Embeddings t-SNE Visualization ({len(self.documents)} chunks)"
        if self.tsne_results.shape[1] == 2:
            fig.update_layout(
                title=title,
                xaxis_title="t-SNE Component 1",
                yaxis_title="t-SNE Component 2",
                hovermode='closest',
                width=1200,
                height=800,
                template='plotly_white'
            )
        else:
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="t-SNE Component 1",
                    yaxis_title="t-SNE Component 2",
                    zaxis_title="t-SNE Component 3",
                ),
                width=1200,
                height=800,
                template='plotly_white'
            )
        
        # Add click event to show full text
        fig.update_layout(
            clickmode='event+select',
            showlegend=True if color_by else False,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            )
        )
        
        return fig
    
    def save_visualization(self, fig: go.Figure, output_file: str = "embeddings_visualization.html"):
        """Save the interactive plot to an HTML file."""
        # Use Plotly's built-in HTML generation with custom config
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
        }
        
        # Simple HTML template without customdata handling
        html_string = '''
        <html>
        <head>
            <meta charset="utf-8" />
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
            </style>
        </head>
        <body>
            {plotly_html}
        </body>
        </html>
        '''
        
        # Generate the Plotly HTML
        plotly_html = fig.to_html(
            include_plotlyjs='cdn',
            config=config,
            div_id="myplot"
        )
        
        # Combine with our custom HTML
        full_html = html_string.format(plotly_html=plotly_html)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"Visualization saved to {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(description="Visualize document embeddings using t-SNE")
    parser.add_argument("--index", default="document_index.json", 
                        help="Path to document index file")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="t-SNE perplexity parameter")
    parser.add_argument("--dimensions", type=int, default=2, choices=[2, 3],
                        help="Number of t-SNE dimensions")
    parser.add_argument("--color-by", default="file", choices=["file", "chunk_index", "none"],
                        help="How to color the points")
    parser.add_argument("--output", default="embeddings_visualization.html",
                        help="Output HTML file")
    parser.add_argument("--no-cache", action="store_true",
                        help="Don't use cached t-SNE results")
    parser.add_argument("--point-size", type=int, default=8,
                        help="Size of scatter points")
    parser.add_argument("--opacity", type=float, default=0.7,
                        help="Opacity of points")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = EmbeddingVisualizer(args.index)
    
    # Load index
    if not visualizer.load_index():
        sys.exit(1)
    
    # Compute t-SNE
    cache_file = None if args.no_cache else f"tsne_cache_{args.dimensions}d.npy"
    visualizer.compute_tsne(
        n_components=args.dimensions,
        perplexity=args.perplexity,
        cache_file=cache_file
    )
    
    # Create visualization
    print("Creating interactive visualization...")
    color_by = None if args.color_by == "none" else args.color_by
    fig = visualizer.create_interactive_plot(
        color_by=color_by,
        point_size=args.point_size,
        opacity=args.opacity
    )
    
    # Save visualization
    output_file = visualizer.save_visualization(fig, args.output)
    
    print(f"\nVisualization complete! Open {output_file} in your browser to explore.")
    print("\nInteraction tips:")
    print("- Hover over points to see document details")
    print("- Click on a point to view the full text")
    print("- Use the legend to toggle file groups on/off")
    print("- Zoom and pan to explore different regions")


if __name__ == "__main__":
    main()