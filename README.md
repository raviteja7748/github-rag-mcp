# 🚀 GitHub RAG MCP Server

**Transform any GitHub repository into a powerful RAG (Retrieval-Augmented Generation) system with intelligent caching and real-time progress tracking.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://platform.openai.com/)
[![Supabase](https://img.shields.io/badge/Supabase-Database-orange.svg)](https://supabase.com/)
[![MCP](https://img.shields.io/badge/MCP-Compatible-purple.svg)](https://modelcontextprotocol.io/)

## 🎯 What This Does

This MCP (Model Context Protocol) server **directly clones GitHub repositories** and transforms them into structured RAG systems that can be queried by AI assistants. Perfect for code analysis, documentation search, and understanding large codebases.

### ✨ Key Features

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

## 🚀 Quick Start

### 1. **Clone & Install**

   ```bash
git clone https://github.com/your-username/github-rag-mcp.git
cd github-rag-mcp
pip install -r requirements.txt
```

### 2. **Database Setup**

1. Create a [Supabase](https://supabase.com) project
2. Run the SQL schema from `sql/github_rag_schema.sql`

### 3. **Configuration**

Create a `.env` file:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
SUMMARY_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Supabase Configuration  
SUPABASE_URL=your_supabase_project_url_here
SUPABASE_SERVICE_KEY=your_supabase_service_key_here

# Performance Features
CACHE_ENABLED=true
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_RERANKING=true
```

### 4. **Start the Server**

```bash
python src/github_rag_mcp.py
```

Server runs on `http://localhost:8052`

### 5. **Configure Your AI Assistant**

#### For Cursor IDE:
Add to your MCP settings:
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
