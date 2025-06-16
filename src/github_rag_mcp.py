"""
MCP server for GitHub repository RAG processing.

This server provides tools to clone GitHub repositories, chunk code and documentation,
generate summaries and embeddings, and upload structured data to Supabase for RAG workflows.
"""
from mcp.server.fastmcp import FastMCP, Context
from sentence_transformers import CrossEncoder
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv
from supabase import Client
from pathlib import Path
import asyncio
import json
import os
import re
import concurrent.futures
import tempfile
import shutil

from git import Repo
from git.exc import GitCommandError

from utils import (
    get_supabase_client,
    add_code_chunks_to_supabase,
    add_doc_chunks_to_supabase,
    search_code_chunks,
    search_doc_chunks,
    chunk_code_file,
    chunk_documentation,
    generate_code_summary,
    generate_doc_summary,
    create_embeddings_batch,
    update_repository_info,
    extract_repository_summary,
    generate_contextual_embedding,
    process_chunk_with_context
)

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Create a dataclass for our application context
@dataclass
class GitHubRAGContext:
    """Context for the GitHub RAG MCP server."""
    supabase_client: Client
    reranking_model: Optional[CrossEncoder] = None

@asynccontextmanager
async def github_rag_lifespan(server: FastMCP) -> AsyncIterator[GitHubRAGContext]:
    """
    Manages the GitHub RAG client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        GitHubRAGContext: The context containing Supabase client and reranking model
    """
    # Initialize Supabase client
    supabase_client = get_supabase_client()
    
    # Initialize cross-encoder model for reranking if enabled
    reranking_model = None
    if os.getenv("USE_RERANKING", "false") == "true":
        try:
            reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"Failed to load reranking model: {e}")
            reranking_model = None
    
    try:
        yield GitHubRAGContext(
            supabase_client=supabase_client,
            reranking_model=reranking_model
        )
    finally:
        # Cleanup if needed
        pass

# Initialize FastMCP server
mcp = FastMCP(
    "github-rag-mcp",
    description="MCP server for GitHub repository RAG processing",
    lifespan=github_rag_lifespan
)

# Health check can be done through the MCP tools if needed

def rerank_results(model: CrossEncoder, query: str, results: List[Dict[str, Any]], content_key: str = "content") -> List[Dict[str, Any]]:
    """
    Rerank search results using a cross-encoder model.
    
    Args:
        model: The cross-encoder model to use for reranking
        query: The search query
        results: List of search results
        content_key: The key in each result dict that contains the text content
        
    Returns:
        Reranked list of results
    """
    if not model or not results:
        return results
    
    try:
        # Extract content from results
        texts = [result.get(content_key, "") for result in results]
        
        # Create pairs of [query, document] for the cross-encoder
        pairs = [[query, text] for text in texts]
        
        # Get relevance scores from the cross-encoder
        scores = model.predict(pairs)
        
        # Add scores to results and sort by score (descending)
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return reranked
    except Exception as e:
        print(f"Error during reranking: {e}")
        return results

def is_code_file(file_path: str) -> bool:
    """
    Check if a file is a code file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a code file, False otherwise
    """
    code_extensions = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
        '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.clj',
        '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd', '.sql', '.r',
        '.m', '.mm', '.pl', '.pm', '.lua', '.dart', '.elm', '.ex', '.exs',
        '.fs', '.fsx', '.ml', '.mli', '.hs', '.lhs', '.nim', '.cr', '.d',
        '.jl', '.zig', '.v', '.vv', '.odin', '.pony', '.red', '.rkt', '.scm',
        '.ss', '.sls', '.lisp', '.lsp', '.cl', '.el', '.cljs', '.cljc'
    }
    return Path(file_path).suffix.lower() in code_extensions

def is_documentation_file(file_path: str) -> bool:
    """
    Check if a file is a documentation file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a documentation file, False otherwise
    """
    doc_extensions = {'.md', '.rst', '.txt', '.adoc', '.asciidoc'}
    doc_names = {'readme', 'changelog', 'license', 'contributing', 'authors', 'install', 'usage'}
    
    path = Path(file_path)
    
    # Check extension
    if path.suffix.lower() in doc_extensions:
        return True
    
    # Check filename without extension
    if path.stem.lower() in doc_names:
        return True
    
    return False

def should_skip_file(file_path: str) -> bool:
    """
    Check if a file should be skipped during processing.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file should be skipped, False otherwise
    """
    skip_patterns = {
        # Version control
        '.git', '.svn', '.hg',
        # Dependencies
        'node_modules', '__pycache__', '.pytest_cache', 'venv', '.venv', 'env', '.env',
        'vendor', 'target', 'build', 'dist', '.next', '.nuxt',
        # IDE/Editor files
        '.vscode', '.idea', '.vs', '*.swp', '*.swo', '*~',
        # OS files
        '.DS_Store', 'Thumbs.db', 'desktop.ini',
        # Logs and temp files
        '*.log', '*.tmp', '*.temp', '*.cache',
        # Binary files
        '*.exe', '*.dll', '*.so', '*.dylib', '*.bin', '*.obj', '*.o',
        # Images and media
        '*.png', '*.jpg', '*.jpeg', '*.gif', '*.svg', '*.ico', '*.webp',
        '*.mp4', '*.avi', '*.mov', '*.wmv', '*.flv', '*.webm',
        '*.mp3', '*.wav', '*.flac', '*.aac', '*.ogg',
        # Archives
        '*.zip', '*.tar', '*.gz', '*.bz2', '*.xz', '*.7z', '*.rar',
        # Package files
        '*.jar', '*.war', '*.ear', '*.deb', '*.rpm', '*.msi', '*.dmg',
        # Lock files (but keep package.json, requirements.txt, etc.)
        'package-lock.json', 'yarn.lock', 'Pipfile.lock', 'poetry.lock', 'Cargo.lock'
    }
    
    path = Path(file_path)
    
    # Check if any part of the path matches skip patterns
    for part in path.parts:
        if part in skip_patterns:
            return True
        # Check wildcard patterns
        for pattern in skip_patterns:
            if '*' in pattern:
                import fnmatch
                if fnmatch.fnmatch(part, pattern):
                    return True
    
    # Smart file size handling - don't skip important large code files
    try:
        if path.exists():
            file_size = path.stat().st_size
            
            # Skip extremely large files (>10MB) - likely data files or binaries
            if file_size > 10 * 1024 * 1024:
                return True
            
            # For files between 1MB-10MB, only skip if they're not important code/docs
            if file_size > 1024 * 1024:
                # Don't skip important code files even if large
                if is_code_file(str(path)) or is_documentation_file(str(path)):
                    print(f"Processing large file: {file_path} ({file_size / 1024 / 1024:.1f}MB)")
                    return False  # Process large code/doc files
                else:
                    return True   # Skip large non-code files
    except:
        pass
    
    return False 

@mcp.tool()
async def clone_and_process_repository(ctx: Context, repo_url: str, branch: str = "main") -> str:
    """
    Clone a GitHub repository and process it into structured RAG data.
    
    This tool clones the specified repository, analyzes all code and documentation files,
    chunks them appropriately, generates summaries and embeddings, and stores everything
    in Supabase for retrieval-augmented generation workflows.
    
    Args:
        ctx: The MCP server provided context
        repo_url: URL of the GitHub repository to clone and process
        branch: Branch to clone (default: "main")
    
    Returns:
        JSON string with processing summary and statistics
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Parse repository information
        parsed_url = urlparse(repo_url)
        if 'github.com' not in parsed_url.netloc:
            return json.dumps({
                "success": False,
                "error": "Only GitHub repositories are supported"
            }, indent=2)
        
        # Extract owner and repo name
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 2:
            return json.dumps({
                "success": False,
                "error": "Invalid GitHub repository URL format"
            }, indent=2)
        
        owner = path_parts[0]
        repo_name = path_parts[1].replace('.git', '')
        repo_id = f"{owner}/{repo_name}"
        
        # Create temporary directory for cloning
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / repo_name
            
            try:
                # Clone the repository
                print(f"Cloning repository: {repo_url}")
                repo = Repo.clone_from(repo_url, repo_path, branch=branch)
                
                # Get repository metadata
                try:
                    commit_hash = repo.head.commit.hexsha[:8]
                    commit_message = repo.head.commit.message.strip()
                    commit_date = repo.head.commit.committed_datetime.isoformat()
                except:
                    commit_hash = "unknown"
                    commit_message = "Unable to retrieve commit info"
                    commit_date = "unknown"
                
            except GitCommandError as e:
                return json.dumps({
                    "success": False,
                    "error": f"Failed to clone repository: {str(e)}"
                }, indent=2)
            
            # Process files
            code_files = []
            doc_files = []
            skipped_files = 0
            
            for file_path in repo_path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(repo_path)
                    
                    if should_skip_file(str(relative_path)):
                        skipped_files += 1
                        continue
                    
                    try:
                        # Try to read the file as text
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        
                        # Skip empty files or files that are too small
                        if len(content.strip()) < 50:
                            continue
                        
                        file_info = {
                            'path': str(relative_path),
                            'content': content,
                            'size': len(content)
                        }
                        
                        if is_code_file(str(relative_path)):
                            code_files.append(file_info)
                        elif is_documentation_file(str(relative_path)):
                            doc_files.append(file_info)
                        
                    except Exception as e:
                        print(f"Error reading file {relative_path}: {e}")
                        continue
            
            # Process code files
            code_chunks_data = []
            if code_files:
                print(f"Processing {len(code_files)} code files...")
                
                for file_info in code_files:
                    chunks = chunk_code_file(file_info['content'], file_info['path'])
                    
                    for i, chunk in enumerate(chunks):
                        code_chunks_data.append({
                            'repo_id': repo_id,
                            'file_path': file_info['path'],
                            'chunk_index': i,
                            'content': chunk['content'],
                            'metadata': {
                                'file_type': 'code',
                                'language': chunk.get('language', 'unknown'),
                                'start_line': chunk.get('start_line', 0),
                                'end_line': chunk.get('end_line', 0),
                                'char_count': len(chunk['content']),
                                'word_count': len(chunk['content'].split())
                            }
                        })
            
            # Process documentation files
            doc_chunks_data = []
            if doc_files:
                print(f"Processing {len(doc_files)} documentation files...")
                
                for file_info in doc_files:
                    chunks = chunk_documentation(file_info['content'], file_info['path'])
                    
                    for i, chunk in enumerate(chunks):
                        doc_chunks_data.append({
                            'repo_id': repo_id,
                            'file_path': file_info['path'],
                            'chunk_index': i,
                            'content': chunk['content'],
                            'metadata': {
                                'file_type': 'documentation',
                                'headers': chunk.get('headers', ''),
                                'char_count': len(chunk['content']),
                                'word_count': len(chunk['content'].split())
                            }
                        })
            
            # Generate summaries and embeddings with progress tracking
            total_chunks = len(code_chunks_data) + len(doc_chunks_data)
            print(f"📊 Processing {total_chunks} chunks total...")
            print(f"   📝 Code chunks: {len(code_chunks_data)}")
            print(f"   📚 Doc chunks: {len(doc_chunks_data)}")
            
            # Check if contextual embeddings are enabled
            use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
            print(f"🔗 Using contextual embeddings: {use_contextual_embeddings}")
            
            # Create file content mapping for contextual embeddings
            file_content_map = {}
            if use_contextual_embeddings:
                for file_info in code_files + doc_files:
                    file_content_map[file_info['path']] = file_info['content']
            
            # Process code chunks with progress tracking
            if code_chunks_data:
                print(f"🔄 Step 1/4: Generating summaries for {len(code_chunks_data)} code chunks...")
                
                # Generate summaries for code chunks
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    code_summary_futures = []
                    for chunk_data in code_chunks_data:
                        future = executor.submit(
                            generate_code_summary,
                            chunk_data['content'],
                            chunk_data['file_path'],
                            chunk_data['metadata'].get('language', 'unknown')
                        )
                        code_summary_futures.append(future)
                    
                    # Collect summaries with progress
                    completed = 0
                    for i, future in enumerate(code_summary_futures):
                        try:
                            summary = future.result()
                            code_chunks_data[i]['summary'] = summary
                            completed += 1
                            if completed % 10 == 0 or completed == len(code_summary_futures):
                                print(f"   ✅ Generated {completed}/{len(code_summary_futures)} code summaries")
                        except Exception as e:
                            print(f"❌ Error generating summary for code chunk {i}: {e}")
                            code_chunks_data[i]['summary'] = "Code implementation"
                
                print(f"🔄 Step 2/4: Generating embeddings for {len(code_chunks_data)} code chunks...")
                
                # Generate embeddings for code chunks (with contextual embeddings if enabled)
                if use_contextual_embeddings:
                    # Import the contextual embedding functions
                    from utils import process_chunk_with_context
                    
                    print("   🔗 Processing contextual embeddings...")
                    # Prepare arguments for contextual embedding processing
                    contextual_args = []
                    for chunk_data in code_chunks_data:
                        full_document = file_content_map.get(chunk_data['file_path'], '')
                        chunk_content = f"{chunk_data['summary']}\n\n{chunk_data['content']}"
                        contextual_args.append((full_document, chunk_content, chunk_data['file_path']))
                    
                    # Process contextual embeddings in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        contextual_results = list(executor.map(process_chunk_with_context, contextual_args))
                    
                    # Extract contextual texts and create embeddings
                    contextual_texts = [result[0] for result in contextual_results]
                    code_embeddings = create_embeddings_batch(contextual_texts)
                    
                    # Update metadata to indicate contextual embedding was used
                    for i, (contextual_text, success) in enumerate(contextual_results):
                        code_chunks_data[i]['metadata']['contextual_embedding'] = success
                else:
                    # Standard embedding without context
                    code_texts = [f"{chunk['summary']}\n\n{chunk['content']}" for chunk in code_chunks_data]
                    code_embeddings = create_embeddings_batch(code_texts)
                
                for i, embedding in enumerate(code_embeddings):
                    code_chunks_data[i]['embedding'] = embedding
                
                print(f"   💾 Storing {len(code_chunks_data)} code chunks in database...")
                # Store code chunks in Supabase
                add_code_chunks_to_supabase(supabase_client, code_chunks_data)
            
            # Process documentation chunks with progress tracking
            if doc_chunks_data:
                print(f"🔄 Step 3/4: Generating summaries for {len(doc_chunks_data)} documentation chunks...")
                
                # Generate summaries for documentation chunks
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    doc_summary_futures = []
                    for chunk_data in doc_chunks_data:
                        future = executor.submit(
                            generate_doc_summary,
                            chunk_data['content'],
                            chunk_data['file_path']
                        )
                        doc_summary_futures.append(future)
                    
                    # Collect summaries with progress
                    completed = 0
                    for i, future in enumerate(doc_summary_futures):
                        try:
                            summary = future.result()
                            doc_chunks_data[i]['summary'] = summary
                            completed += 1
                            if completed % 5 == 0 or completed == len(doc_summary_futures):
                                print(f"   ✅ Generated {completed}/{len(doc_summary_futures)} doc summaries")
                        except Exception as e:
                            print(f"❌ Error generating summary for doc chunk {i}: {e}")
                            doc_chunks_data[i]['summary'] = "Documentation content"
                
                print(f"🔄 Step 4/4: Generating embeddings for {len(doc_chunks_data)} documentation chunks...")
                
                # Generate embeddings for documentation chunks (with contextual embeddings if enabled)
                if use_contextual_embeddings:
                    # Import the contextual embedding functions
                    from utils import process_chunk_with_context
                    
                    print("   🔗 Processing contextual embeddings...")
                    # Prepare arguments for contextual embedding processing
                    contextual_args = []
                    for chunk_data in doc_chunks_data:
                        full_document = file_content_map.get(chunk_data['file_path'], '')
                        chunk_content = f"{chunk_data['summary']}\n\n{chunk_data['content']}"
                        contextual_args.append((full_document, chunk_content, chunk_data['file_path']))
                    
                    # Process contextual embeddings in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        contextual_results = list(executor.map(process_chunk_with_context, contextual_args))
                    
                    # Extract contextual texts and create embeddings
                    contextual_texts = [result[0] for result in contextual_results]
                    doc_embeddings = create_embeddings_batch(contextual_texts)
                    
                    # Update metadata to indicate contextual embedding was used
                    for i, (contextual_text, success) in enumerate(contextual_results):
                        doc_chunks_data[i]['metadata']['contextual_embedding'] = success
                else:
                    # Standard embedding without context
                    doc_texts = [f"{chunk['summary']}\n\n{chunk['content']}" for chunk in doc_chunks_data]
                    doc_embeddings = create_embeddings_batch(doc_texts)
                
                for i, embedding in enumerate(doc_embeddings):
                    doc_chunks_data[i]['embedding'] = embedding
                
                print(f"   💾 Storing {len(doc_chunks_data)} documentation chunks in database...")
                # Store documentation chunks in Supabase
                add_doc_chunks_to_supabase(supabase_client, doc_chunks_data)
            
            # Generate repository summary
            readme_content = ""
            for doc_file in doc_files:
                if 'readme' in doc_file['path'].lower():
                    readme_content = doc_file['content'][:5000]  # First 5000 chars
                    break
            
            repo_summary = extract_repository_summary(repo_id, readme_content)
            total_word_count = sum(chunk['metadata']['word_count'] for chunk in code_chunks_data + doc_chunks_data)
            
            # Update repository information
            update_repository_info(
                supabase_client,
                repo_id,
                repo_summary,
                total_word_count,
                len(code_files),
                len(doc_files),
                commit_hash,
                commit_message,
                commit_date,
                branch
            )
            
            return json.dumps({
                "success": True,
                "repository": {
                    "id": repo_id,
                    "url": repo_url,
                    "branch": branch,
                    "commit_hash": commit_hash,
                    "commit_message": commit_message,
                    "commit_date": commit_date
                },
                "processing_stats": {
                    "code_files_processed": len(code_files),
                    "documentation_files_processed": len(doc_files),
                    "code_chunks_created": len(code_chunks_data),
                    "documentation_chunks_created": len(doc_chunks_data),
                    "total_chunks": total_chunks,
                    "total_word_count": total_word_count,
                    "files_skipped": skipped_files
                },
                "summary": repo_summary
            }, indent=2)
            
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def get_available_repositories(ctx: Context) -> str:
    """
    Get all available repositories from the database.
    
    This tool returns a list of all repositories that have been processed and stored
    in the database, along with their summaries and statistics. Use this to discover
    what repositories are available for querying.
    
    Args:
        ctx: The MCP server provided context
    
    Returns:
        JSON string with the list of available repositories and their details
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Query the repositories table
        result = supabase_client.from_('repositories')\
            .select('*')\
            .order('repo_id')\
            .execute()
        
        # Format the repositories with their details
        repositories = []
        if result.data:
            for repo in result.data:
                repositories.append({
                    "repo_id": repo.get("repo_id"),
                    "summary": repo.get("summary"),
                    "total_word_count": repo.get("total_word_count"),
                    "code_files_count": repo.get("code_files_count"),
                    "doc_files_count": repo.get("doc_files_count"),
                    "commit_hash": repo.get("commit_hash"),
                    "commit_message": repo.get("commit_message"),
                    "commit_date": repo.get("commit_date"),
                    "branch": repo.get("branch"),
                    "created_at": repo.get("created_at"),
                    "updated_at": repo.get("updated_at")
                })
        
        return json.dumps({
            "success": True,
            "repositories": repositories,
            "count": len(repositories)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def search_code_chunks(ctx: Context, query: str, repo_id: str = None, match_count: int = 5) -> str:
    """
    Search for code chunks relevant to the query.
    
    This tool searches the vector database for code chunks relevant to the query and returns
    the matching code with their summaries. Optionally filter by repository ID.
    
    Args:
        ctx: The MCP server provided context
        query: The search query
        repo_id: Optional repository ID to filter results (e.g., 'owner/repo')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        JSON string with the search results
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
        
        # Prepare filter if repo_id is provided
        filter_metadata = None
        if repo_id and repo_id.strip():
            filter_metadata = {"repo_id": repo_id}
        
        # Perform search
        results = search_code_chunks(
            client=supabase_client,
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata,
            use_hybrid_search=use_hybrid_search
        )
        
        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        if use_reranking and ctx.request_context.lifespan_context.reranking_model:
            results = rerank_results(
                ctx.request_context.lifespan_context.reranking_model, 
                query, 
                results, 
                content_key="content"
            )
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "repo_id": result.get("repo_id"),
                "file_path": result.get("file_path"),
                "content": result.get("content"),
                "summary": result.get("summary"),
                "metadata": result.get("metadata"),
                "similarity": result.get("similarity")
            }
            # Include rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)
        
        return json.dumps({
            "success": True,
            "query": query,
            "repo_filter": repo_id,
            "search_mode": "hybrid" if use_hybrid_search else "vector",
            "reranking_applied": use_reranking and ctx.request_context.lifespan_context.reranking_model is not None,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def search_documentation_chunks(ctx: Context, query: str, repo_id: str = None, match_count: int = 5) -> str:
    """
    Search for relevant documentation chunks using semantic search.
    
    Args:
        query: The search query
        repo_id: Optional repository ID to filter results (get from get_available_repositories)
        match_count: Number of results to return (default: 5, max: 20)
        
    Returns:
        JSON string containing search results with content, metadata, and similarity scores
    """
    try:
        context = ctx.get_context(GitHubRAGContext)
        
        # Validate match_count
        match_count = min(max(match_count, 1), 20)
        
        # Prepare filter if repo_id is provided
        filter_metadata = None
        if repo_id and repo_id.strip():
            filter_metadata = {"repo_id": repo_id}
        
        # Search for documentation chunks
        results = search_doc_chunks(
            client=context.supabase_client,
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata,
            use_hybrid_search=False
        )
        
        # Apply reranking if enabled
        if context.reranking_model and results:
            results = rerank_results(context.reranking_model, query, results, "content")
        
        return json.dumps({
            "query": query,
            "repo_filter": repo_id,
            "total_results": len(results),
            "results": results
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"Failed to search documentation chunks: {str(e)}",
            "query": query,
            "repo_filter": repo_id
        }, indent=2)

@mcp.tool()
async def get_repository_details(ctx: Context, repo_id: str) -> str:
    """
    Get detailed information about a specific repository.
    
    Args:
        repo_id: The repository ID to get details for
        
    Returns:
        JSON string containing repository details including stats and metadata
    """
    try:
        context = ctx.get_context(GitHubRAGContext)
        
        # Get repository info
        repo_response = context.supabase_client.table("repositories").select("*").eq("id", repo_id).execute()
        
        if not repo_response.data:
            return json.dumps({
                "error": f"Repository with ID {repo_id} not found"
            }, indent=2)
        
        repo_info = repo_response.data[0]
        
        # Get code chunks count
        code_count_response = context.supabase_client.table("code_chunks").select("id", count="exact").eq("repo_id", repo_id).execute()
        code_chunks_count = code_count_response.count or 0
        
        # Get doc chunks count
        doc_count_response = context.supabase_client.table("doc_chunks").select("id", count="exact").eq("repo_id", repo_id).execute()
        doc_chunks_count = doc_count_response.count or 0
        
        # Get language distribution for code chunks
        lang_response = context.supabase_client.table("code_chunks").select("language").eq("repo_id", repo_id).execute()
        languages = {}
        for chunk in lang_response.data:
            lang = chunk.get("language", "unknown")
            languages[lang] = languages.get(lang, 0) + 1
        
        return json.dumps({
            "repository": repo_info,
            "statistics": {
                "code_chunks": code_chunks_count,
                "doc_chunks": doc_chunks_count,
                "total_chunks": code_chunks_count + doc_chunks_count,
                "languages": languages
            }
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"Failed to get repository details: {str(e)}",
            "repo_id": repo_id
        }, indent=2)

@mcp.tool()
async def delete_repository(ctx: Context, repo_id: str) -> str:
    """
    Delete a repository and all its associated chunks from the database.
    
    Args:
        repo_id: The repository ID to delete
        
    Returns:
        JSON string confirming deletion or error message
    """
    try:
        context = ctx.get_context(GitHubRAGContext)
        
        # Check if repository exists
        repo_response = context.supabase_client.table("repositories").select("name, url").eq("id", repo_id).execute()
        
        if not repo_response.data:
            return json.dumps({
                "error": f"Repository with ID {repo_id} not found"
            }, indent=2)
        
        repo_info = repo_response.data[0]
        
        # Get counts before deletion
        code_count_response = context.supabase_client.table("code_chunks").select("id", count="exact").eq("repo_id", repo_id).execute()
        code_chunks_count = code_count_response.count or 0
        
        doc_count_response = context.supabase_client.table("doc_chunks").select("id", count="exact").eq("repo_id", repo_id).execute()
        doc_chunks_count = doc_count_response.count or 0
        
        # Delete repository (cascading deletes will handle chunks)
        delete_response = context.supabase_client.table("repositories").delete().eq("id", repo_id).execute()
        
        return json.dumps({
            "success": True,
            "message": f"Successfully deleted repository '{repo_info['name']}'",
            "deleted_repository": repo_info,
            "deleted_chunks": {
                "code_chunks": code_chunks_count,
                "doc_chunks": doc_chunks_count,
                "total": code_chunks_count + doc_chunks_count
            }
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"Failed to delete repository: {str(e)}",
            "repo_id": repo_id
        }, indent=2)

@mcp.tool()
async def update_repository_metadata(ctx: Context, repo_id: str, description: str = None, tags: List[str] = None) -> str:
    """
    Update repository metadata like description and tags.
    
    Args:
        repo_id: The repository ID to update
        description: Optional new description for the repository
        tags: Optional list of tags to associate with the repository
        
    Returns:
        JSON string confirming update or error message
    """
    try:
        context = ctx.get_context(GitHubRAGContext)
        
        # Check if repository exists
        repo_response = context.supabase_client.table("repositories").select("*").eq("id", repo_id).execute()
        
        if not repo_response.data:
            return json.dumps({
                "error": f"Repository with ID {repo_id} not found"
            }, indent=2)
        
        # Prepare update data
        update_data = {}
        if description is not None:
            update_data["description"] = description
        if tags is not None:
            update_data["tags"] = tags
        
        if not update_data:
            return json.dumps({
                "error": "No update data provided. Specify description and/or tags."
            }, indent=2)
        
        # Update repository
        update_response = context.supabase_client.table("repositories").update(update_data).eq("id", repo_id).execute()
        
        return json.dumps({
            "success": True,
            "message": "Repository metadata updated successfully",
            "repo_id": repo_id,
            "updated_fields": update_data,
            "repository": update_response.data[0] if update_response.data else None
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"Failed to update repository metadata: {str(e)}",
            "repo_id": repo_id
        }, indent=2)

async def main():
    """Main function to run the MCP server."""
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport (uses default host and port)
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main()) 