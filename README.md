# 🚀 GitHub RAG MCP Server

**Transform any GitHub repository into a powerful, searchable knowledge base with intelligent caching and real-time progress tracking.**

## 📋 Project Summary

This MCP (Model Context Protocol) server converts GitHub repositories into structured RAG (Retrieval-Augmented Generation) systems that AI assistants can query intelligently. Unlike simple file access, this tool processes repositories into semantic chunks with AI-generated summaries and vector embeddings, enabling deep code understanding and efficient search capabilities.

**Key capabilities:**
- 🔄 **Direct repository processing**: Clone, analyze, and transform any GitHub repo
- 🧠 **AI-powered understanding**: GPT-4 summaries for every code chunk  
- 🔍 **Semantic search**: Find code by meaning, not just keywords
- 💾 **Smart caching**: 90% faster subsequent runs with intelligent cache system
- 📊 **Real-time progress**: Live updates during processing

## 🆚 GitHub MCP Comparison

| Feature | Official GitHub MCP | This GitHub RAG MCP |
|---------|-------------------|-------------------|
| **Data Access** | Raw file content via GitHub API | Processed, chunked, and summarized content |
| **Search** | File path and name searching | Semantic search with AI embeddings |
| **Understanding** | Basic file reading | AI-generated summaries for each code chunk |
| **Performance** | API rate limited | Cached processing with 90% speed improvement |
| **Offline Access** | Requires GitHub API | Local database with offline search |
| **Use Case** | Quick file access | Deep code analysis and understanding |

**When to use this over official GitHub MCP:**
- ✅ Analyzing large, complex codebases
- ✅ Finding code by functionality, not just file names  
- ✅ Getting AI-powered insights into repository structure
- ✅ Working with repositories repeatedly (caching benefits)
- ✅ Semantic code search and discovery

## 🛠️ Installation

### Prerequisites

- **Python 3.12+** 
- **OpenAI API key** (for embeddings and summaries)
- **Supabase account** (free tier works)

### Method 1: Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is the fastest Python package manager (10-100x faster than pip) and provides better dependency resolution.

#### **Step 1: Install uv** 

**Don't have uv yet? Choose the best method for your system:**

**🚀 Option A: Standalone Installer (Recommended - No Python required)**
```bash
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**🐍 Option B: Using pip (if you already have Python)**
```bash
pip install uv
```

**📦 Option C: Using pipx (if you have pipx)**
```bash
pipx install uv
```

**🍺 Option D: Package Managers**
```bash
# macOS (Homebrew)
brew install uv

# Windows (Scoop)
scoop install uv

# Windows (Chocolatey)
choco install uv
```

#### **Step 2: Setup Project**

```bash
# 1. Clone the repository
git clone https://github.com/raviteja7748/github-rag-mcp.git
cd github-rag-mcp

# 2. Install dependencies (uv automatically creates virtual environment)
uv sync

# 3. Activate the environment (optional - uv run works without activation)
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### Method 2: Using pip

```bash
# 1. Clone the repository
git clone https://github.com/raviteja7748/github-rag-mcp.git
cd github-rag-mcp

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

### Database Setup

1. **Create Supabase Project**
   - Go to [supabase.com](https://supabase.com) and create a new project
   - Wait for the project to be ready (usually 1-2 minutes)

2. **Run Database Schema**
   - Open the SQL Editor in your Supabase dashboard
   - Copy and paste the contents of `sql/github_rag_schema.sql`
   - Execute the SQL to create the required tables and functions

3. **Get Credentials**
   - Go to Settings → API in your Supabase dashboard
   - Copy your Project URL and service_role API key

### Environment Configuration

Create a `.env` file in the project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
SUMMARY_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Supabase Configuration  
SUPABASE_URL=your_supabase_project_url_here
SUPABASE_SERVICE_KEY=your_supabase_service_key_here

# Performance Features (Optional)
CACHE_ENABLED=true
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_RERANKING=true
```

### Start the Server

```bash
# Using uv
uv run python src/github_rag_mcp.py

# Using pip (with activated environment)
python src/github_rag_mcp.py
```

The server will start on `http://localhost:8052`

### Configure Your AI Assistant

#### For Cursor IDE:
Add to your MCP settings (`.cursor/mcp.json`):
```json
{
  "mcpServers": {
    "github-rag": {
      "transport": "sse",
      "url": "http://localhost:8052/sse"
    }
  }
}
```

#### For Claude Desktop:
Add to your Claude configuration:
```json
{
  "mcpServers": {
    "github-rag": {
      "command": "python",
      "args": ["path/to/github-rag-mcp/src/github_rag_mcp.py"]
    }
  }
}
```

## ✨ Key Features

- 🔄 **Direct GitHub Integration**: Clone and process any public repository
- 🧠 **AI-Powered Analysis**: GPT-4 summaries for every code chunk
- 🔍 **Semantic Search**: OpenAI embeddings for intelligent code discovery  
- 💾 **Smart Caching**: 90% faster subsequent runs, massive API cost savings
- 📊 **Real-time Progress**: Live updates during processing with emoji indicators
- 🏗️ **Intelligent Chunking**: Code by functions/classes, docs by headers
- 🔒 **Secure Processing**: Temporary directories, automatic cleanup
- 🎯 **Cursor Integration**: Native MCP support for seamless workflow

## 🔥 Performance Benefits

### **Caching System**
- **First run**: Full processing with OpenAI API calls
- **Subsequent runs**: ~90% cache hits = 10x faster processing
- **Cost savings**: Dramatically reduces OpenAI API usage
- **Smart recovery**: Automatic cache corruption handling

### **Processing Speed**
- **Small repos** (<100 files): ~30 seconds
- **Medium repos** (100-1000 files): ~2-5 minutes  
- **Large repos** (1000+ files): ~10-30 minutes
- **Cache-enabled reruns**: ~80% faster

## 🎬 Live Processing Example

When you process a repository, you'll see real-time progress:

```
📊 Step 1/4: Cloning repository...
   ✅ Cloned avijeett007/kno2gether-webrtc-agent (develop branch)

🔄 Step 2/4: Processing 156 files...
   📝 Code chunks: 128
   📚 Doc chunks: 28

🔄 Step 3/4: Generating summaries...
   ✅ Generated 10/128 code summaries
   ✅ Generated 50/128 code summaries
   📡 Creating 15 new embeddings (found 113 in cache)
   ✅ All summaries complete!

🔄 Step 4/4: Storing in database...
   💾 Stored 128 code chunks
   💾 Stored 28 documentation chunks
   ✅ Processing complete!
```

## 🛠️ Available Tools

### **Repository Processing**
- **`clone_and_process_repository`**: Transform any GitHub repo into searchable RAG data
- **`get_available_repositories`**: List all processed repositories with stats

### **Intelligent Search**
- **`search_code_chunks`**: Find relevant code with AI-generated summaries
- **`search_documentation_chunks`**: Search docs with semantic similarity scoring

### **Repository Management**
- **`get_repository_details`**: Detailed statistics and metadata
- **`delete_repository`**: Clean removal with cascading deletes
- **`update_repository_metadata`**: Update descriptions and tags

## 🔍 Supported Content

### **Code Files (40+ Languages)**
Python, JavaScript, TypeScript, Java, C/C++, C#, PHP, Ruby, Go, Rust, Swift, Kotlin, Scala, Shell scripts, SQL, R, Lua, Dart, and many more...

### **Documentation Files**
Markdown (`.md`), reStructuredText (`.rst`), Plain text (`.txt`), AsciiDoc (`.adoc`), README files, CHANGELOG, LICENSE, etc.

### **Automatically Skipped**
- **Dependencies**: `node_modules`, `__pycache__`, `venv`, `vendor`
- **Version control**: `.git`, `.svn`, `.hg`  
- **Build artifacts**: `build`, `dist`, `target`, `.next`
- **Binary files**: Images, executables, archives
- **Large files**: >10MB (with smart handling for important code files)

## 🔒 Privacy & Security

- **🗂️ Temporary Processing**: Repositories cloned to temp directories, auto-deleted after processing
- **🔐 No File Storage**: Only processed chunks and metadata stored in database
- **🛡️ Secure Credentials**: All API keys via environment variables
- **🏠 Local Processing**: All file analysis happens on your machine
- **🧹 Automatic Cleanup**: Zero traces left on your system

## 📊 Database Schema

### **Core Tables**
- **`repositories`**: Repository metadata, stats, and processing info
- **`code_chunks`**: Code segments with embeddings and AI summaries  
- **`doc_chunks`**: Documentation chunks with semantic embeddings

### **Search Functions**
- **`match_code_chunks()`**: Vector similarity search for code
- **`match_doc_chunks()`**: Vector similarity search for documentation

## ⚙️ Configuration Options

### **Caching Control**
```env
CACHE_ENABLED=true          # Enable/disable intelligent caching
```

### **AI Model Selection**
```env
SUMMARY_MODEL=gpt-4o-mini              # For generating code summaries
EMBEDDING_MODEL=text-embedding-3-small # For creating vector embeddings
```

### **Advanced RAG Features**
```env
USE_CONTEXTUAL_EMBEDDINGS=true    # Better search accuracy
USE_HYBRID_SEARCH=true            # Vector + keyword search
USE_RERANKING=true                # Improved result ordering
```

## 🎯 Use Cases

- **📚 Code Documentation**: Understand large, unfamiliar codebases
- **🔍 Code Search**: Find specific functions, patterns, or implementations
- **🤖 AI Assistant Enhancement**: Give your AI deep knowledge of any repository
- **📖 Learning**: Study open-source projects with AI-powered insights
- **🔧 Development**: Quick reference and code discovery during development

## 🏗️ Architecture

```
GitHub Repo → Clone → Process → Chunk → Summarize → Embed → Store → Search
     ↓           ↓        ↓        ↓         ↓        ↓       ↓
   Public     Temp     Smart   AI-Gen    OpenAI   Supabase  MCP
   Repos      Dir    Chunking Summary  Embeddings Database Tools
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the [Model Context Protocol (MCP)](https://modelcontextprotocol.io)
- Powered by [OpenAI](https://openai.com/) embeddings and GPT models
- Uses [Supabase](https://supabase.com/) for vector storage and search
- Inspired by modern RAG architectures and intelligent caching strategies

---

**Ready to transform any GitHub repository into a powerful, searchable knowledge base? Get started now!** 🚀 
