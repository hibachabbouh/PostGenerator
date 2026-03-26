# 📸 PostGenerator - InstaGen AI

An intelligent Instagram caption generator powered by RAG (Retrieval-Augmented Generation) and LLM technology. Generate creative, engaging captions tailored to your post's topic and desired style.

## ✨ Features

- **RAG-Powered Generation**: Uses semantic search to find similar high-quality captions and generates contextually relevant ones
- **Multiple Styles**: Generate captions in different tones:
  - ✨ Motivational
  - 😂 Funny
  - 🎨 Aesthetic
  - 📝 General
  - 🔥 Trendy
- **Real-time Streaming**: Captions are generated and displayed in real-time
- **Intelligent Retrieval**: Smart semantic search with FAISS for retrieving similar examples
- **API-First Design**: RESTful FastAPI backend for easy integration
- **Modern UI**: Streamlit-based frontend with Instagram-inspired styling
- **Docker Ready**: Complete Docker setup for easy deployment

## 🛠️ Technologies

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **LLM**: Groq API (LLaMA models)
- **ML/Embeddings**: Sentence Transformers, FAISS
- **Data Processing**: Pandas, Datasets
- **Containerization**: Docker, Docker Compose

## 📋 Requirements

- Python 3.8+
- Groq API key (get one at https://console.groq.com)
- Docker & Docker Compose (for containerized setup)

## 🚀 Installation

### Local Setup

1. **Clone the repository** (or extract the project directory)

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Prepare the data** (first time only):
   ```bash
   python data/load_dataset.py
   python data/preprocess.py
   python ml/create_embeddings.py
   ```

## 📖 Usage

### Running Locally

**Terminal 1 - Start the backend API**:
```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`
- Health check: `http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`

**Terminal 2 - Start the frontend**:
```bash
streamlit run frontend/streamlit_app.py
```

The app will be available at `http://localhost:8501`

### Using the Application

1. Open the Streamlit app in your browser
2. Enter your post topic (e.g., "A sunset at the beach in Tunisia")
3. Select a caption style/vibe
4. Click "Generate Caption"
5. View the generated caption and generation history

### Docker Setup

Run everything with Docker Compose:

```bash
docker-compose up
```

- Backend API: `http://localhost:8000`
- Streamlit app: `http://localhost:8501`

To rebuild images:
```bash
docker-compose up --build
```

## 📁 Project Structure

```
PostGenerator/
├── app/
│   ├── main.py              # FastAPI application
│   ├── routes/
│   │   └── generate.py      # Caption generation endpoint
│   └── services/
│       └── rag_pipeline.py  # RAG pipeline logic
├── frontend/
│   └── streamlit_app.py     # Streamlit UI
├── ml/
│   ├── create_embeddings.py # Embedding generation
│   └── retriever.py         # FAISS-powered retriever
├── data/
│   ├── load_dataset.py      # Dataset loading
│   ├── preprocess.py        # Data preprocessing
│   ├── raw/                 # Raw data files
│   └── processed/           # Processed/enriched data
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile.backend       # Backend container
├── Dockerfile.frontend      # Frontend container
└── requirements.txt         # Python dependencies
```

## 🔌 API Endpoints

### Health Check
```
GET /health
```
Returns: `{"status": "ok"}`

### Generate Caption
```
POST /api/generate
Content-Type: application/json

{
    "topic": "A sunset at the beach",
    "style": "✨ Motivational"
}
```

**Response**:
```json
{
    "caption": "Generated caption text here...",
    "style": "✨ Motivational",
    "examples_used": [...]
}
```

Access the interactive API documentation at `http://localhost:8000/docs` when the backend is running.

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
API_BASE_URL=http://127.0.0.1:8000  # Backend URL for frontend
STREAMLIT_THEME=light               # Streamlit theme
```

## 🧠 How It Works

1. **Retrieval**: User provides a topic and style
2. **Search**: Smart retriever finds k=5 most similar captions from the database
3. **Generation**: Groq LLM generates new caption based on retrieved examples
4. **Display**: Caption is streamed and displayed in real-time

The RAG approach ensures:
- Generated captions are contextually relevant
- Style is consistent with examples
- High engagement factor (uses real examples as reference)

## 📊 Data Pipeline

1. **Load**: Captions loaded from social media dataset
2. **Preprocess**: Clean text, extract features (tone, hook strength)
3. **Embed**: Generate vector embeddings using Sentence Transformers
4. **Index**: Build FAISS index for fast similarity search
5. **Enrich**: LLM enriches captions with metadata

## 🐳 Docker Deployment

### Build Individual Images

**Backend**:
```bash
docker build -f Dockerfile.backend -t postgenerator-backend .
```

**Frontend**:
```bash
docker build -f Dockerfile.frontend -t postgenerator-frontend .
```

### Production Deployment

For production, consider:
- Using environment-specific `.env` files
- SSL/TLS certificates for API endpoints
- Load balancing for API instances
- Persistent volume for embeddings and indexes

## 🧪 Development

### Adding New Styles

1. Add style to `STYLES` in `frontend/streamlit_app.py`
2. Update retriever to handle new style in `ml/retriever.py`
3. Test with the generate endpoint

### Updating the Dataset

1. Update data files in `data/raw/`
2. Run preprocessing: `python data/preprocess.py`
3. Recreate embeddings: `python ml/create_embeddings.py`
4. Restart the application

## 👤 Author

**Hiba Chabbouh**

- GitHub: [@hibachabbouh](https://github.com/hibachabbouh)
- LinkedIn: [Hiba Chabbouh](https://www.linkedin.com/in/hiba-chabbouh/)
