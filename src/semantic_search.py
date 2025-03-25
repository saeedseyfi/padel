import logging
import os
import json
import glob
import numpy as np
import argparse
from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple
import time
from dotenv import load_dotenv
import pickle
import hashlib
import sys
from pathlib import Path
import warnings
from cohere.client_v2 import ClientV2

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

logger = logging.getLogger(__name__)


class PadelSemanticSearch:
    def __init__(self, resources_dir=None, cache_dir=None):
        """
        Initialize the Padel semantic search engine.
        
        Args:
            resources_dir: Directory containing the video metadata JSON files
            cache_dir: Directory to store embedding cache files
        """        
        # Use default paths relative to project root if not specified
        self.resources_dir = resources_dir or os.path.join(PROJECT_ROOT, "resources/videos")
        self.cache_dir = cache_dir or os.path.join(PROJECT_ROOT, "cache")
        self.sections = []
        self.metadata = []
        self.section_embeddings = None
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.cohere_client = ClientV2(api_key=os.environ.get("COHERE_API_KEY"))
            
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimensions = 612
        self.rerank_model = "rerank-v3.5"
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "sections"), exist_ok=True)
        
        # Load all video metadata
        self._load_metadata()
        
        # Load or create embeddings section by section
        self._load_or_create_embeddings()
        
    def _get_section_hash(self, section_text: str) -> str:
        """Generate a hash for a specific section text."""
        return hashlib.md5(section_text.encode()).hexdigest()
    
    def _get_section_cache_path(self, section_hash: str) -> str:
        """Get the cache file path for a specific section."""
        return os.path.join(
            self.cache_dir, 
            "sections",
            f"{section_hash}_{self.embedding_model}.pkl"
        )
    
    def _get_metadata_cache_path(self) -> str:
        """Get the cache file path for metadata mapping."""
        return os.path.join(self.cache_dir, "metadata.json")
    
    def _load_section_embedding(self, section_text: str) -> Optional[np.ndarray]:
        """Load embedding for a single section if it exists in cache."""
        section_hash = self._get_section_hash(section_text)
        cache_path = self._get_section_cache_path(section_hash)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Verify the cached embedding is compatible
                if (cache_data.get('model') == self.embedding_model and
                    cache_data.get('dimensions') == self.embedding_dimensions):
                    return np.array(cache_data['embedding'])
            except Exception as e:
                print(f"Error loading section cache: {e}")
        
        return None
    
    def _save_section_embedding(self, section_text: str, embedding: np.ndarray):
        """Save embedding for a single section to cache."""
        section_hash = self._get_section_hash(section_text)
        cache_path = self._get_section_cache_path(section_hash)
        
        try:
            cache_data = {
                'model': self.embedding_model,
                'dimensions': self.embedding_dimensions,
                'timestamp': time.time(),
                'embedding': embedding.tolist()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Error saving section cache: {e}")
    
    def _save_metadata_mapping(self):
        """Save metadata mapping to track which sections are in the dataset."""
        cache_path = self._get_metadata_cache_path()
        
        try:
            # Create a mapping of section hashes to indexes
            section_mapping = {
                self._get_section_hash(text): idx 
                for idx, text in enumerate(self.sections)
            }
            
            metadata = {
                'model': self.embedding_model,
                'dimensions': self.embedding_dimensions,
                'timestamp': time.time(),
                'section_count': len(self.sections),
                'section_mapping': section_mapping
            }
            
            with open(cache_path, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            print(f"Error saving metadata mapping: {e}")
    
    def _get_sections_to_update(self) -> Tuple[List[int], List[int]]:
        """
        Determine which sections need to be updated based on cache.
        
        Returns:
            Tuple containing:
                - List of indexes for sections that need embedding updates
                - List of indexes for sections that can be loaded from cache
        """
        # Check if metadata mapping exists
        cache_path = self._get_metadata_cache_path()
        old_mapping = {}
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache_metadata = json.load(f)
                    
                # Check if the cache is compatible with current settings
                if (cache_metadata.get('model') == self.embedding_model and
                    cache_metadata.get('dimensions') == self.embedding_dimensions):
                    old_mapping = cache_metadata.get('section_mapping', {})
            except Exception as e:
                logger.error(f"Error loading metadata mapping: {e}")
        
        # Determine which sections need to be updated
        sections_to_update = []
        sections_from_cache = []
        
        for idx, section_text in enumerate(self.sections):
            section_hash = self._get_section_hash(section_text)
            cache_path = self._get_section_cache_path(section_hash)
            
            # Check if this section exists in cache
            if section_hash in old_mapping and os.path.exists(cache_path):
                sections_from_cache.append(idx)
            else:
                sections_to_update.append(idx)
        
        return sections_to_update, sections_from_cache
    
    def _load_or_create_embeddings(self):
        """Load embeddings from cache or create new ones for sections that have changed."""
        if not self.sections:
            logger.error("No sections loaded. Cannot create embeddings.")
            return
        
        logger.debug(f"Processing embeddings for {len(self.sections)} sections...")
        
        # Determine which sections need to be updated
        sections_to_update, sections_from_cache = self._get_sections_to_update()
        
        logger.debug(f"Found {len(sections_from_cache)} sections in cache, {len(sections_to_update)} require updating.")
        
        # Initialize the embeddings array
        self.section_embeddings = np.zeros((len(self.sections), self.embedding_dimensions))
        
        # Load cached sections
        for idx in sections_from_cache:
            section_text = self.sections[idx]
            embedding = self._load_section_embedding(section_text)
            if embedding is not None:
                self.section_embeddings[idx] = embedding
        
        # Update sections that need new embeddings
        if sections_to_update:
            self._create_embeddings_for_sections(sections_to_update)
            
        # Save metadata mapping for future reference
        self._save_metadata_mapping()
    
    def _create_embeddings_for_sections(self, section_indices: List[int]):
        """Create embeddings for specific sections and update the embeddings array."""
        if not section_indices:
            return
            
        logger.debug(f"Creating embeddings for {len(section_indices)} sections...")
        
        # Process in batches to avoid rate limits
        batch_size = 100
        section_indices_batches = [
            section_indices[i:i+batch_size] 
            for i in range(0, len(section_indices), batch_size)
        ]
        
        for batch_num, batch_indices in enumerate(section_indices_batches):
            logger.debug(f"Processing batch {batch_num+1}/{len(section_indices_batches)}...")
            
            # Get the text for this batch
            batch_texts = [self.sections[idx] for idx in batch_indices]
            
            try:
                # Create embeddings for the batch
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts,
                    dimensions=self.embedding_dimensions
                )
                
                # Process each embedding in the batch
                for i, embedding_data in enumerate(response.data):
                    section_idx = batch_indices[i]
                    embedding = np.array(embedding_data.embedding)
                    
                    # Store the embedding in the array
                    self.section_embeddings[section_idx] = embedding
                    
                    # Cache the individual section embedding
                    self._save_section_embedding(self.sections[section_idx], embedding)
                
                # Be nice to the API rate limits
                if batch_num < len(section_indices_batches) - 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error creating embeddings for batch {batch_num+1}: {e}")
                # Continue with whatever embeddings we have
                break
        
        logger.debug(f"Completed embedding updates for {len(section_indices)} sections.")
    
    def _load_metadata(self):
        """Load all video metadata from JSON files."""
        json_files = glob.glob(os.path.join(self.resources_dir, "*.json"))
        
        logger.debug(f"Loading metadata from {len(json_files)} files...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    video_data = json.load(f)
                    
                # Video level metadata
                video_id = os.path.basename(json_file).replace('.json', '')
                title = video_data.get('title', '')
                url = video_data.get('url', '')
                topics = video_data.get('topics', [])
                notes = video_data.get('notes', '')
                
                # Process each section in the video
                if 'sections' in video_data:
                    for section in video_data['sections']:
                        section_title = section.get('title', '')
                        description = section.get('description', '')
                        start_time = section.get('start_time', 0)
                        end_time = section.get('end_time', 0)
                        
                        # Create search text that combines all relevant information
                        search_text = f"{title} - {section_title}: {description}"
                        
                        # Add topics to enrich search context
                        if topics:
                            search_text += f" Topics: {', '.join(topics)}"
                        
                        # Add video notes if available
                        if notes:
                            search_text += f" {notes}"
                            
                        # Store the section info and metadata
                        self.sections.append(search_text)
                        
                        # Store metadata for retrieval
                        self.metadata.append({
                            'video_id': video_id,
                            'title': title,
                            'description': description,
                            'section_title': section_title,
                            'url': url,
                            'start_time': start_time,
                            'end_time': end_time,
                            'topics': topics
                        })
                        
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                
        logger.debug(f"Loaded {len(self.sections)} sections from {len(json_files)} videos.")
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search(self, query, top_k=5):
        """
        Search for sections relevant to the query using OpenAI embeddings.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing search results with metadata
        """
        if self.section_embeddings is None or len(self.section_embeddings) == 0:
            logger.error("No embeddings available. Cannot perform search.")
            return []
            
        # Get embedding for the query
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[query],
                dimensions=self.embedding_dimensions
            )
            query_embedding = np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error creating query embedding: {e}")
            return []
        
        # Calculate cosine similarities
        similarities = [
            self._cosine_similarity(query_embedding, section_embedding) 
            for section_embedding in self.section_embeddings
        ]
        
        # Get top results
        top_results = []
        top_indices = np.argsort(similarities)[::-1]
        
        for idx in top_indices:
            score = similarities[idx]
                
            if len(top_results) >= top_k:
                break
                
            # Get metadata for this result
            metadata = self.metadata[idx]
            
            # Create YouTube timestamp URL
            video_url = metadata['url']
            timestamp_url = f"{video_url}&t={metadata['start_time']}"
                
            # Format duration as mm:ss
            duration_sec = metadata['end_time'] - metadata['start_time']
            minutes = duration_sec // 60
            seconds = duration_sec % 60
            duration = f"{minutes:02d}:{seconds:02d}"
            
            # Build result object
            result = {
                'score': score,
                'video_id': metadata['video_id'],
                'title': metadata['title'],
                'description': metadata['description'],
                'section_title': metadata['section_title'],
                'url': timestamp_url,
                'start_time': metadata['start_time'],
                'end_time': metadata['end_time'],
                'duration': duration,
                'topics': metadata['topics'],
                'text': self.sections[idx]  # Include the full text for reranking
            }
            
            # Add file_path if available
            if 'file_path' in metadata:
                result['file_path'] = metadata['file_path']
                
            top_results.append(result)
            
        return top_results
        
    def rerank_results(self, results, query, top_k=5):
        """
        Rerank results using Cohere's reranking API.
        
        Args:
            results: The initial search results
            query: Original search query
            top_k: Maximum number of results to return after reranking
            
        Returns:
            Reranked results list
        """
        if not results:
            return []
        
        
        try:
            # Extract documents for reranking
            docs = [result['text'] for result in results]
            
            rerank_response = self.cohere_client.rerank(
                model=self.rerank_model,
                query=query,
                documents=docs,
                top_n=top_k
            )
            
            # Extract reranked indices and relevance scores
            reranked_results = []
            for rank_result in rerank_response.results:
                idx = rank_result.index
                results[idx]['rerank_score'] = rank_result.relevance_score
                reranked_results.append(results[idx])
            
            # Sort by rerank score
            reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
            return reranked_results
            
        except Exception:
            logger.exception(f"Error during reranking")
            logger.error("Falling back to original ranking...")
            return results[:top_k]

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def group_results_by_file(resources_dir, results, top_k=None):
    """
    Group search results by JSON file path and collect the actual section data.
    
    Args:
        resources_dir: Directory containing the video metadata JSON files
        results: The search results to organize
        top_k: Number of files to return (limits the return to top-k files)
        
    Returns:
        List of dictionaries, each with 'file' and 'sections' keys
    """
    # First group by file
    file_groups = {}
    for result in results:
        file_path = f"{resources_dir}/{result['video_id']}.json"
        
        if file_path not in file_groups:
            file_groups[file_path] = {
                'sections': [],
                'max_score': 0  # Track the max score for this file
            }
            
        # Create a section object with the actual data
        section = {
            'title': result['section_title'],
            'description': result['description'],
            'start_time': result['start_time'],
            'end_time': result['end_time'],
            'url': result['url'],
            'score': result.get('rerank_score', result.get('score', 0)),
        }
        
        file_groups[file_path]['sections'].append(section)
        
        # Update the max score for this file if this section has a higher score
        file_groups[file_path]['max_score'] = max(
            file_groups[file_path]['max_score'], 
            section['score']
        )
    
    # Sort files by their highest scoring section
    sorted_files = sorted(
        file_groups.items(), 
        key=lambda x: x[1]['max_score'], 
        reverse=True
    )
    
    # Limit to top_k files if specified
    if top_k is not None:
        sorted_files = sorted_files[:top_k]
    
    # Convert to the desired output format
    grouped_data = []
    for file_path, file_data in sorted_files:
        # Sort sections within each file by score (highest first)
        file_data['sections'].sort(key=lambda x: x['score'], reverse=True)

        try:
            file = load_json(file_path)
            title = file.get('title', 'Unknown Title')
            url = file.get('url', '')
            topics = file.get('topics', [])
            description = file.get('notes', '')

            grouped_data.append({
                'title': title,
                'description': description,
                'url': url,
                'topics': topics,
                'sections': file_data['sections']
            })
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
    
    return grouped_data

def main():
    parser = argparse.ArgumentParser(description="Semantic search for Padel video sections")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of files to show")
    parser.add_argument("--rerank", action="store_true", default=True, help="Apply Cohere reranking")
    parser.add_argument("--rerank-model", type=str, help="Specify a rerank model (e.g., 'rerank-v3.5', 'rerank-english-v2.0')")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()

    load_dotenv()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    
    # Get many more results than we need for filtering by file later
    # We need to retrieve enough sections to have good coverage of files
    section_limit = args.top_k * 3
    
    # Initialize search engine
    search_engine = PadelSemanticSearch()
    
    # Set custom rerank model if specified
    if args.rerank_model:
        search_engine.rerank_model = args.rerank_model
        logger.debug(f"Using custom rerank model: {args.rerank_model}")
    
    # Perform search with higher section limit
    results = search_engine.search(args.query, section_limit)
    
    # Apply reranking if specified and we have results
    if args.rerank and results:
        results = search_engine.rerank_results(results, args.query, section_limit)
    
    # Display results
    if results:
        logger.debug(f"\nFound {len(results)} relevant sections for '{args.query}':\n")
        
        # Group results and limit to top_k files
        grouped_results = group_results_by_file(search_engine.resources_dir, results, args.top_k)
        
        logger.debug(f"Returning top {len(grouped_results)} files")
        
        for grouped_result in grouped_results:
            # Output each result as a complete JSON object
            print(json.dumps(grouped_result))
    else:
        logger.debug(f"No results found for '{args.query}'")

if __name__ == "__main__":
    main()