
# DMRC HR Assistant - Web Application

A Flask-based web application that integrates with the DMRC chatbot RAG (Retrieval-Augmented Generation) system to provide an interactive HR assistant for Delhi Metro Rail Corporation.

## Features

- 🚇 **DMRC HR Assistant**: Interactive chatbot for HR policies and procedures
- 🤖 **RAG System**: Uses LangChain + ChromaDB + Groq API for intelligent responses
- 💬 **Modern Web Interface**: Clean, responsive chat interface
- 📚 **Source Citations**: Shows relevant document sources for each response
- ⚡ **Real-time Chat**: Instant responses with loading indicators

## Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended for embeddings)
- Groq API key
- HuggingFace access token

## Setup Instructions

### 1. Environment Setup

```bash
# Clone/navigate to the project directory
cd "c:\Users\Samarth\OneDrive\Desktop\dmrc chatbot"

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the `.env.sample` file in the `DMRC_Chatbot/` directory to `.env` and update with your API keys:

```bash
cd DMRC_Chatbot
cp .env.sample .env
```

Edit `.env` file:
```
HF_ACCESS_KEY=your_huggingface_token_here
TOGETHER_API_KEY=your_together_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Verify Required Files

Ensure these files exist in the `DMRC_Chatbot/` directory:
- `chunks.json` (1MB+ file containing processed document chunks)
- `.env` (with your API keys)
- `src/langchain_rag.py` (RAG system implementation)

### 4. Run the Application

```bash
# From the main directory (where app.py is located)
python app.py
```

The application will:
1. Initialize the RAG system (may take 2-3 minutes on first run)
2. Build/load the ChromaDB vector store
3. Start the Flask web server on `http://localhost:5000`

## Usage

1. **Open your browser** and navigate to `http://localhost:5000`
2. **Wait for initialization** - the system status indicator will turn green when ready
3. **Start chatting** - Ask questions about HR policies, procedures, guidelines, etc.

### Example Questions

- "What are the employee leave policies?"
- "How does the performance evaluation process work?"
- "What is the dress code policy?"
- "Tell me about employee benefits"
- "What are the working hours and attendance rules?"

## API Endpoints

- `GET /` - Main chat interface
- `POST /api/chat` - Send chat messages
- `GET /api/status` - Get system status
- `GET /api/test` - Test RAG system functionality

## Project Structure

```
dmrc chatbot/
├── app.py                 # Flask web application
├── templates/
│   └── index.html         # Chat interface template
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── DMRC_Chatbot/         # Core RAG system
    ├── chunks.json       # Processed document chunks
    ├── .env              # Environment variables
    ├── src/
    │   ├── langchain_rag.py    # RAG implementation
    │   ├── chunker.py          # Document processing
    │   └── embedder.py         # Embedding utilities
    └── hr_chroma_db_web/ # Vector database (created automatically)
```

## Troubleshooting

### System Not Initializing

1. **Check API Keys**: Ensure `GROQ_API_KEY` is set in `DMRC_Chatbot/.env`
2. **Verify Files**: Ensure `chunks.json` exists and is not empty
3. **Check Logs**: Look at console output for specific error messages
4. **GPU Issues**: If CUDA errors occur, the system will fall back to CPU

### No Responses from Chatbot

1. **Test System**: Visit `/api/test` to check if documents are found
2. **Rebuild Vector Store**: Delete `hr_chroma_db_web/` folder and restart
3. **Check Network**: Ensure internet connection for Groq API calls

### Performance Issues

1. **GPU Acceleration**: Ensure CUDA is properly installed
2. **Memory**: Close other applications if running out of RAM
3. **Model Loading**: First run takes longer due to model downloads

## Development

To modify the chatbot behavior:

1. **RAG System**: Edit `DMRC_Chatbot/src/langchain_rag.py`
2. **Web Interface**: Edit `templates/index.html` or `app.py`
3. **Styling**: Modify CSS in `templates/index.html`

## License

This project is for internal DMRC use. Please ensure compliance with organizational policies regarding AI and data usage.

## Support

For technical issues:
1. Check the console logs for detailed error messages
2. Verify all dependencies are installed correctly
3. Ensure API keys are valid and have sufficient credits
4. Test individual components using the provided test endpoints

