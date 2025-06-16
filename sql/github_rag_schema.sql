-- Enable the pgvector extension to work with embedding vectors
CREATE EXTENSION IF NOT EXISTS vector;

-- Create repositories table to store repository metadata
CREATE TABLE IF NOT EXISTS repositories (
    id BIGSERIAL PRIMARY KEY,
    repo_id TEXT UNIQUE NOT NULL, -- Format: "owner/repo"
    summary TEXT,
    total_word_count INTEGER DEFAULT 0,
    code_files_count INTEGER DEFAULT 0,
    doc_files_count INTEGER DEFAULT 0,
    commit_hash TEXT,
    commit_message TEXT,
    commit_date TIMESTAMP,
    branch TEXT DEFAULT 'main',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create code_chunks table to store code chunks with embeddings
CREATE TABLE IF NOT EXISTS code_chunks (
    id BIGSERIAL PRIMARY KEY,
    repo_id TEXT NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    metadata JSONB,
    embedding VECTOR(1536), -- OpenAI text-embedding-3-small dimension
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(repo_id, file_path, chunk_index)
);

-- Create doc_chunks table to store documentation chunks with embeddings
CREATE TABLE IF NOT EXISTS doc_chunks (
    id BIGSERIAL PRIMARY KEY,
    repo_id TEXT NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    metadata JSONB,
    embedding VECTOR(1536), -- OpenAI text-embedding-3-small dimension
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(repo_id, file_path, chunk_index)
);

-- Essential indexes
CREATE INDEX IF NOT EXISTS idx_repositories_repo_id ON repositories(repo_id);
CREATE INDEX IF NOT EXISTS idx_code_chunks_repo_id ON code_chunks(repo_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_repo_id ON doc_chunks(repo_id);
CREATE INDEX IF NOT EXISTS idx_code_chunks_embedding ON code_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding ON doc_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Search function for code chunks
CREATE OR REPLACE FUNCTION match_code_chunks(
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 5,
    filter JSONB DEFAULT '{}'::jsonb
)
RETURNS TABLE(
    id BIGINT,
    repo_id TEXT,
    file_path TEXT,
    chunk_index INTEGER,
    content TEXT,
    summary TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id, c.repo_id, c.file_path, c.chunk_index, c.content, c.summary, c.metadata,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM code_chunks c
    WHERE (filter = '{}'::jsonb OR (filter ? 'repo_id' AND c.repo_id = filter->>'repo_id'))
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Search function for documentation chunks
CREATE OR REPLACE FUNCTION match_doc_chunks(
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 5,
    filter JSONB DEFAULT '{}'::jsonb
)
RETURNS TABLE(
    id BIGINT,
    repo_id TEXT,
    file_path TEXT,
    chunk_index INTEGER,
    content TEXT,
    summary TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id, d.repo_id, d.file_path, d.chunk_index, d.content, d.summary, d.metadata,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM doc_chunks d
    WHERE (filter = '{}'::jsonb OR (filter ? 'repo_id' AND d.repo_id = filter->>'repo_id'))
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;