"""
Utility functions for the GitHub RAG MCP server.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import json
from supabase import create_client, Client
from urllib.parse import urlparse
import openai
import re
import time
import hashlib
import pickle
from pathlib import Path

# Load OpenAI API key for embeddings
openai.api_key = os.getenv("OPENAI_API_KEY")

# Simple file-based cache
class SimpleCache:
    """Simple file-based cache for embeddings and summaries."""
    
    def __init__(self, cache_dir: str = "./cache_data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    
    def _get_cache_key(self, text: str, cache_type: str, model: str = "") -> str:
        """Generate cache key from text content."""
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"{cache_type}_{model}_{content_hash}.pkl"
    
    def get(self, text: str, cache_type: str, model: str = "") -> Optional[Any]:
        """Get cached item if exists."""
        if not self.enabled:
            return None
            
        cache_file = self.cache_dir / self._get_cache_key(text, cache_type, model)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error reading cache file {cache_file}: {e}")
                # Remove corrupted cache file
                try:
                    cache_file.unlink()
                except:
                    pass
        return None
    
    def set(self, text: str, cache_type: str, value: Any, model: str = "") -> None:
        """Cache item."""
        if not self.enabled:
            return
            
        cache_file = self.cache_dir / self._get_cache_key(text, cache_type, model)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Error writing cache file {cache_file}: {e}")
    
    def get_batch_embeddings(self, texts: List[str], model: str) -> Tuple[Dict[int, List[float]], List[int]]:
        """Get cached embeddings and return (cached_embeddings, missing_indices)."""
        cached_embeddings = {}
        missing_indices = []
        
        for i, text in enumerate(texts):
            embedding = self.get(text, "embedding", model)
            if embedding:
                cached_embeddings[i] = embedding
            else:
                missing_indices.append(i)
        
        return cached_embeddings, missing_indices

# Global cache instance
_cache = SimpleCache()

def get_supabase_client() -> Client:
    """
    Get a Supabase client with the URL and key from environment variables.
    
    Returns:
        Supabase client instance
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables")
    
    return create_client(url, key)

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts with caching support.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
    
    # Get embedding model from environment
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Check cache first
    cached_embeddings, missing_indices = _cache.get_batch_embeddings(texts, embedding_model)
    
    # If all embeddings are cached, return them
    if not missing_indices:
        print(f"✅ All {len(texts)} embeddings found in cache!")
        return [cached_embeddings[i] for i in range(len(texts))]
    
    # Only call API for missing embeddings
    print(f"📡 Creating {len(missing_indices)} new embeddings (found {len(cached_embeddings)} in cache)")
    missing_texts = [texts[i] for i in missing_indices]
    
    max_retries = 3
    retry_delay = 1.0
    
    for retry in range(max_retries):
        try:
            response = openai.embeddings.create(
                model=embedding_model,
                input=missing_texts
            )
            new_embeddings = [item.embedding for item in response.data]
            
            # Cache new embeddings
            for i, embedding in zip(missing_indices, new_embeddings):
                _cache.set(texts[i], "embedding", embedding, embedding_model)
                cached_embeddings[i] = embedding
            
            # Return all embeddings in original order
            return [cached_embeddings[i] for i in range(len(texts))]
            
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Failed to create batch embeddings after {max_retries} attempts: {e}")
                # Try creating embeddings one by one as fallback
                print("Attempting to create embeddings individually...")
                
                for i in missing_indices:
                    try:
                        individual_response = openai.embeddings.create(
                            model=embedding_model,
                            input=[texts[i]]
                        )
                        embedding = individual_response.data[0].embedding
                        _cache.set(texts[i], "embedding", embedding, embedding_model)
                        cached_embeddings[i] = embedding
                    except Exception as individual_error:
                        print(f"Failed to create embedding for text {i}: {individual_error}")
                        # Add zero embedding as fallback
                        cached_embeddings[i] = [0.0] * 1536
                
                return [cached_embeddings[i] for i in range(len(texts))]

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using OpenAI's API.
    
    Args:
        text: Text to create an embedding for
        
    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        return [0.0] * 1536

def chunk_code_file(content: str, file_path: str, max_chunk_size: int = 2000) -> List[Dict[str, Any]]:
    """
    Chunk a code file into meaningful segments.
    
    Args:
        content: The file content
        file_path: Path to the file
        max_chunk_size: Maximum size of each chunk in characters
        
    Returns:
        List of chunks with metadata
    """
    chunks = []
    
    # Detect language from file extension
    language = detect_language(file_path)
    
    # Split content into lines
    lines = content.split('\n')
    
    # Try to chunk by functions/classes first
    if language in ['python', 'javascript', 'typescript', 'java', 'cpp', 'c']:
        chunks = chunk_by_functions(lines, language, max_chunk_size)
    
    # If function-based chunking didn't work well, fall back to simple chunking
    if not chunks or len(chunks) == 1:
        chunks = chunk_by_lines(lines, max_chunk_size)
    
    # Add metadata to chunks
    for i, chunk in enumerate(chunks):
        chunk.update({
            'language': language,
            'file_path': file_path,
            'chunk_index': i
        })
    
    return chunks

def chunk_documentation(content: str, file_path: str, max_chunk_size: int = 3000) -> List[Dict[str, Any]]:
    """
    Chunk documentation content by headers and paragraphs.
    
    Args:
        content: The documentation content
        file_path: Path to the file
        max_chunk_size: Maximum size of each chunk in characters
        
    Returns:
        List of chunks with metadata
    """
    chunks = []
    
    # For Markdown files, chunk by headers
    if file_path.endswith('.md'):
        chunks = chunk_markdown_by_headers(content, max_chunk_size)
    else:
        # For other documentation, chunk by paragraphs
        chunks = chunk_by_paragraphs(content, max_chunk_size)
    
    # Add metadata to chunks
    for i, chunk in enumerate(chunks):
        chunk.update({
            'file_path': file_path,
            'chunk_index': i
        })
    
    return chunks

def detect_language(file_path: str) -> str:
    """
    Detect programming language from file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Detected language or 'unknown'
    """
    extension_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.php': 'php',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.sh': 'bash',
        '.bash': 'bash',
        '.sql': 'sql',
        '.r': 'r',
        '.lua': 'lua',
        '.dart': 'dart'
    }
    
    ext = Path(file_path).suffix.lower()
    return extension_map.get(ext, 'unknown')

def chunk_by_functions(lines: List[str], language: str, max_chunk_size: int) -> List[Dict[str, Any]]:
    """
    Chunk code by functions/classes/methods.
    
    Args:
        lines: List of code lines
        language: Programming language
        max_chunk_size: Maximum chunk size
        
    Returns:
        List of chunks
    """
    chunks = []
    current_chunk = []
    current_size = 0
    
    # Define function/class patterns for different languages
    patterns = {
        'python': [r'^\s*def\s+', r'^\s*class\s+', r'^\s*async\s+def\s+'],
        'javascript': [r'^\s*function\s+', r'^\s*const\s+\w+\s*=\s*\(', r'^\s*class\s+'],
        'typescript': [r'^\s*function\s+', r'^\s*const\s+\w+\s*=\s*\(', r'^\s*class\s+'],
        'java': [r'^\s*public\s+.*\s*\{', r'^\s*private\s+.*\s*\{', r'^\s*protected\s+.*\s*\{', r'^\s*class\s+'],
        'cpp': [r'^\s*\w+.*\s*\{', r'^\s*class\s+', r'^\s*struct\s+'],
        'c': [r'^\s*\w+.*\s*\{', r'^\s*struct\s+']
    }
    
    function_patterns = patterns.get(language, [])
    
    for i, line in enumerate(lines):
        line_size = len(line) + 1  # +1 for newline
        
        # Check if this line starts a new function/class
        is_function_start = any(re.match(pattern, line) for pattern in function_patterns)
        
        # If we hit a function start and current chunk is not empty, save it
        if is_function_start and current_chunk and current_size > 0:
            chunks.append({
                'content': '\n'.join(current_chunk),
                'start_line': i - len(current_chunk) + 1,
                'end_line': i
            })
            current_chunk = []
            current_size = 0
        
        # Add line to current chunk
        current_chunk.append(line)
        current_size += line_size
        
        # If chunk is getting too large, split it
        if current_size > max_chunk_size and len(current_chunk) > 1:
            chunks.append({
                'content': '\n'.join(current_chunk[:-1]),
                'start_line': i - len(current_chunk) + 2,
                'end_line': i
            })
            current_chunk = [current_chunk[-1]]
            current_size = len(current_chunk[0]) + 1
    
    # Add remaining chunk
    if current_chunk:
        chunks.append({
            'content': '\n'.join(current_chunk),
            'start_line': len(lines) - len(current_chunk) + 1,
            'end_line': len(lines)
        })
    
    return chunks

def chunk_by_lines(lines: List[str], max_chunk_size: int) -> List[Dict[str, Any]]:
    """
    Simple line-based chunking.
    
    Args:
        lines: List of lines
        max_chunk_size: Maximum chunk size
        
    Returns:
        List of chunks
    """
    chunks = []
    current_chunk = []
    current_size = 0
    
    for i, line in enumerate(lines):
        line_size = len(line) + 1  # +1 for newline
        
        # If adding this line would exceed max size, save current chunk
        if current_size + line_size > max_chunk_size and current_chunk:
            chunks.append({
                'content': '\n'.join(current_chunk),
                'start_line': i - len(current_chunk) + 1,
                'end_line': i
            })
            current_chunk = []
            current_size = 0
        
        current_chunk.append(line)
        current_size += line_size
    
    # Add remaining chunk
    if current_chunk:
        chunks.append({
            'content': '\n'.join(current_chunk),
            'start_line': len(lines) - len(current_chunk) + 1,
            'end_line': len(lines)
        })
    
    return chunks

def chunk_markdown_by_headers(content: str, max_chunk_size: int) -> List[Dict[str, Any]]:
    """
    Chunk Markdown content by headers.
    
    Args:
        content: Markdown content
        max_chunk_size: Maximum chunk size
        
    Returns:
        List of chunks with header information
    """
    chunks = []
    lines = content.split('\n')
    current_chunk = []
    current_size = 0
    current_headers = []
    
    for line in lines:
        line_size = len(line) + 1  # +1 for newline
        
        # Check if this is a header
        header_match = re.match(r'^(#+)\s+(.+)$', line)
        if header_match:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append({
                    'content': '\n'.join(current_chunk),
                    'headers': '; '.join(current_headers)
                })
                current_chunk = []
                current_size = 0
            
            # Update headers
            header_level = len(header_match.group(1))
            header_text = header_match.group(2)
            
            # Remove headers of same or lower level
            current_headers = [h for h in current_headers if h.count('#') < header_level]
            current_headers.append(f"{'#' * header_level} {header_text}")
        
        # Add line to current chunk
        current_chunk.append(line)
        current_size += line_size
        
        # If chunk is getting too large, split it
        if current_size > max_chunk_size and len(current_chunk) > 1:
            chunks.append({
                'content': '\n'.join(current_chunk[:-1]),
                'headers': '; '.join(current_headers)
            })
            current_chunk = [current_chunk[-1]]
            current_size = len(current_chunk[0]) + 1
    
    # Add remaining chunk
    if current_chunk:
        chunks.append({
            'content': '\n'.join(current_chunk),
            'headers': '; '.join(current_headers)
        })
    
    return chunks

def chunk_by_paragraphs(content: str, max_chunk_size: int) -> List[Dict[str, Any]]:
    """
    Chunk content by paragraphs.
    
    Args:
        content: Text content
        max_chunk_size: Maximum chunk size
        
    Returns:
        List of chunks
    """
    chunks = []
    paragraphs = content.split('\n\n')
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        paragraph_size = len(paragraph) + 2  # +2 for double newline
        
        # If adding this paragraph would exceed max size, save current chunk
        if current_size + paragraph_size > max_chunk_size and current_chunk:
            chunks.append({
                'content': '\n\n'.join(current_chunk)
            })
            current_chunk = []
            current_size = 0
        
        current_chunk.append(paragraph)
        current_size += paragraph_size
    
    # Add remaining chunk
    if current_chunk:
        chunks.append({
            'content': '\n\n'.join(current_chunk)
        })
    
    return chunks

def generate_code_summary(code: str, file_path: str, language: str) -> str:
    """
    Generate a summary for a code chunk using GPT with caching.
    
    Args:
        code: The code content
        file_path: Path to the file
        language: Programming language
        
    Returns:
        A summary of what the code does
    """
    # Check cache first
    cache_key = f"{code}|{file_path}|{language}"
    cached_summary = _cache.get(cache_key, "code_summary")
    if cached_summary:
        return cached_summary
    
    summary_model = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")
    
    # Create the prompt
    prompt = f"""<code_file_path>
{file_path}
</code_file_path>

<code_language>
{language}
</code_language>

<code_content>
{code[:2000] if len(code) > 2000 else code}
</code_content>

Based on the code content above, provide a concise summary (2-3 sentences) that describes what this code does, its main functionality, and key components. Focus on the practical purpose and implementation details.
"""
    
    try:
        response = openai.chat.completions.create(
            model=summary_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise code summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        summary = response.choices[0].message.content.strip()
        
        # Cache the summary
        _cache.set(cache_key, "code_summary", summary)
        
        return summary
    
    except Exception as e:
        print(f"Error generating code summary: {e}")
        return f"Code implementation in {language}"

def generate_doc_summary(content: str, file_path: str) -> str:
    """
    Generate a summary for a documentation chunk using GPT with caching.
    
    Args:
        content: The documentation content
        file_path: Path to the file
        
    Returns:
        A summary of the documentation content
    """
    # Check cache first
    cache_key = f"{content}|{file_path}"
    cached_summary = _cache.get(cache_key, "doc_summary")
    if cached_summary:
        return cached_summary
    
    summary_model = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")
    
    # Create the prompt
    prompt = f"""<doc_file_path>
{file_path}
</doc_file_path>

<doc_content>
{content[:2000] if len(content) > 2000 else content}
</doc_content>

Based on the documentation content above, provide a concise summary (2-3 sentences) that describes what this documentation covers, its main topics, and key information. Focus on what users would learn from this content.
"""
    
    try:
        response = openai.chat.completions.create(
            model=summary_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise documentation summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        summary = response.choices[0].message.content.strip()
        
        # Cache the summary
        _cache.set(cache_key, "doc_summary", summary)
        
        return summary
    
    except Exception as e:
        print(f"Error generating doc summary: {e}")
        return "Documentation content"

def add_code_chunks_to_supabase(client: Client, chunks_data: List[Dict[str, Any]], batch_size: int = 20) -> None:
    """
    Add code chunks to the Supabase code_chunks table in batches.
    
    Args:
        client: Supabase client
        chunks_data: List of chunk data dictionaries
        batch_size: Size of each batch for insertion
    """
    if not chunks_data:
        return
    
    # Delete existing records for these repositories
    unique_repo_ids = list(set(chunk['repo_id'] for chunk in chunks_data))
    for repo_id in unique_repo_ids:
        try:
            client.table('code_chunks').delete().eq('repo_id', repo_id).execute()
        except Exception as e:
            print(f"Error deleting existing code chunks for {repo_id}: {e}")
    
    # Process in batches
    total_items = len(chunks_data)
    for i in range(0, total_items, batch_size):
        batch_end = min(i + batch_size, total_items)
        batch_data = []
        
        for j in range(i, batch_end):
            chunk = chunks_data[j]
            batch_data.append({
                'repo_id': chunk['repo_id'],
                'file_path': chunk['file_path'],
                'chunk_index': chunk['chunk_index'],
                'content': chunk['content'],
                'summary': chunk['summary'],
                'metadata': chunk['metadata'],
                'embedding': chunk['embedding']
            })
        
        # Insert batch into Supabase with retry logic
        max_retries = 3
        retry_delay = 1.0
        
        for retry in range(max_retries):
            try:
                client.table('code_chunks').insert(batch_data).execute()
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting code chunk batch (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to insert code chunk batch after {max_retries} attempts: {e}")
        
        print(f"Inserted code chunk batch {i//batch_size + 1} of {(total_items + batch_size - 1)//batch_size}")

def add_doc_chunks_to_supabase(client: Client, chunks_data: List[Dict[str, Any]], batch_size: int = 20) -> None:
    """
    Add documentation chunks to the Supabase doc_chunks table in batches.
    
    Args:
        client: Supabase client
        chunks_data: List of chunk data dictionaries
        batch_size: Size of each batch for insertion
    """
    if not chunks_data:
        return
    
    # Delete existing records for these repositories
    unique_repo_ids = list(set(chunk['repo_id'] for chunk in chunks_data))
    for repo_id in unique_repo_ids:
        try:
            client.table('doc_chunks').delete().eq('repo_id', repo_id).execute()
        except Exception as e:
            print(f"Error deleting existing doc chunks for {repo_id}: {e}")
    
    # Process in batches
    total_items = len(chunks_data)
    for i in range(0, total_items, batch_size):
        batch_end = min(i + batch_size, total_items)
        batch_data = []
        
        for j in range(i, batch_end):
            chunk = chunks_data[j]
            batch_data.append({
                'repo_id': chunk['repo_id'],
                'file_path': chunk['file_path'],
                'chunk_index': chunk['chunk_index'],
                'content': chunk['content'],
                'summary': chunk['summary'],
                'metadata': chunk['metadata'],
                'embedding': chunk['embedding']
            })
        
        # Insert batch into Supabase with retry logic
        max_retries = 3
        retry_delay = 1.0
        
        for retry in range(max_retries):
            try:
                client.table('doc_chunks').insert(batch_data).execute()
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting doc chunk batch (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to insert doc chunk batch after {max_retries} attempts: {e}")
        
        print(f"Inserted doc chunk batch {i//batch_size + 1} of {(total_items + batch_size - 1)//batch_size}")

def search_code_chunks(
    client: Client, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None,
    use_hybrid_search: bool = False
) -> List[Dict[str, Any]]:
    """
    Search for code chunks in Supabase using vector similarity.
    
    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        use_hybrid_search: Whether to use hybrid search
        
    Returns:
        List of matching code chunks
    """
    # Create embedding for the query
    query_embedding = create_embedding(f"Code for {query}")
    
    try:
        # Prepare parameters for the RPC call
        params = {
            'query_embedding': query_embedding,
            'match_count': match_count
        }
        
        # Add filter if provided
        if filter_metadata:
            params['filter'] = filter_metadata
        
        # Execute the search using the match_code_chunks function
        result = client.rpc('match_code_chunks', params).execute()
        
        return result.data if result.data else []
    except Exception as e:
        print(f"Error searching code chunks: {e}")
        return []

def search_doc_chunks(
    client: Client, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None,
    use_hybrid_search: bool = False
) -> List[Dict[str, Any]]:
    """
    Search for documentation chunks in Supabase using vector similarity.
    
    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        use_hybrid_search: Whether to use hybrid search
        
    Returns:
        List of matching documentation chunks
    """
    # Create embedding for the query
    query_embedding = create_embedding(query)
    
    try:
        # Prepare parameters for the RPC call
        params = {
            'query_embedding': query_embedding,
            'match_count': match_count
        }
        
        # Add filter if provided
        if filter_metadata:
            params['filter'] = filter_metadata
        
        # Execute the search using the match_doc_chunks function
        result = client.rpc('match_doc_chunks', params).execute()
        
        return result.data if result.data else []
    except Exception as e:
        print(f"Error searching doc chunks: {e}")
        return []

def update_repository_info(
    client: Client, 
    repo_id: str, 
    summary: str, 
    total_word_count: int,
    code_files_count: int,
    doc_files_count: int,
    commit_hash: str,
    commit_message: str,
    commit_date: str,
    branch: str
) -> None:
    """
    Update or insert repository information in the repositories table.
    
    Args:
        client: Supabase client
        repo_id: The repository ID (owner/repo)
        summary: Summary of the repository
        total_word_count: Total word count for the repository
        code_files_count: Number of code files processed
        doc_files_count: Number of documentation files processed
        commit_hash: Git commit hash
        commit_message: Git commit message
        commit_date: Git commit date
        branch: Git branch
    """
    try:
        # Try to update existing repository
        result = client.table('repositories').update({
            'summary': summary,
            'total_word_count': total_word_count,
            'code_files_count': code_files_count,
            'doc_files_count': doc_files_count,
            'commit_hash': commit_hash,
            'commit_message': commit_message,
            'commit_date': commit_date,
            'branch': branch,
            'updated_at': 'now()'
        }).eq('repo_id', repo_id).execute()
        
        # If no rows were updated, insert new repository
        if not result.data:
            client.table('repositories').insert({
                'repo_id': repo_id,
                'summary': summary,
                'total_word_count': total_word_count,
                'code_files_count': code_files_count,
                'doc_files_count': doc_files_count,
                'commit_hash': commit_hash,
                'commit_message': commit_message,
                'commit_date': commit_date,
                'branch': branch
            }).execute()
            print(f"Created new repository: {repo_id}")
        else:
            print(f"Updated repository: {repo_id}")
            
    except Exception as e:
        print(f"Error updating repository {repo_id}: {e}")

def extract_repository_summary(repo_id: str, readme_content: str, max_length: int = 500) -> str:
    """
    Extract a summary for a repository from its README content using an LLM.
    
    Args:
        repo_id: The repository ID (owner/repo)
        readme_content: The README content to extract a summary from
        max_length: Maximum length of the summary
        
    Returns:
        A summary string
    """
    # Default summary if we can't extract anything meaningful
    default_summary = f"Repository {repo_id}"
    
    if not readme_content or len(readme_content.strip()) == 0:
        return default_summary
    
    # Get the summary model from environment variables
    summary_model = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")
    
    # Limit content length to avoid token limits
    truncated_content = readme_content[:10000] if len(readme_content) > 10000 else readme_content
    
    # Create the prompt for generating the summary
    prompt = f"""<repository_id>
{repo_id}
</repository_id>

<readme_content>
{truncated_content}
</readme_content>

The above content is from the README of the repository '{repo_id}'. Please provide a concise summary (3-5 sentences) that describes what this repository/project is about, its main purpose, and key features. The summary should help understand what the project accomplishes and its intended use.
"""
    
    try:
        # Call the OpenAI API to generate the summary
        response = openai.chat.completions.create(
            model=summary_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise repository summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        # Extract the generated summary
        summary = response.choices[0].message.content.strip()
        
        # Ensure the summary is not too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
            
        return summary
    
    except Exception as e:
        print(f"Error generating summary with LLM for {repo_id}: {e}. Using default summary.")
        return default_summary

def generate_contextual_embedding(full_document: str, chunk: str, file_path: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        file_path: Path to the file for additional context
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    summary_model = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")
    
    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document>
{full_document[:25000]}
</document>

<file_path>
{file_path}
</file_path>

Here is the code/documentation chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Consider:
- What module/class/function this belongs to
- The purpose of this code in the larger codebase
- Key concepts or functionality it implements

Answer only with the succinct context and nothing else."""

        # Call the OpenAI API to generate contextual information
        response = openai.chat.completions.create(
            model=summary_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information for code and documentation chunks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        # Extract the generated context
        context = response.choices[0].message.content.strip()
        
        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"
        
        return contextual_text, True
    
    except Exception as e:
        print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False

def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (full_document, chunk, file_path)
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    full_document, chunk, file_path = args
    return generate_contextual_embedding(full_document, chunk, file_path) 