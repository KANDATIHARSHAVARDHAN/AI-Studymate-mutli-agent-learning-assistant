# Multi-Agentic RAG System 🚀

Welcome to the **Multi-Agentic RAG (Retrieval-Augmented Generation) System**. This project is a highly advanced, intelligent backend and frontend ecosystem that uses dynamic agent routing to parse user queries, fetch relevant documents, and synthesize highly accurate, specialized responses. 

## 🏗️ Architecture & Workflow

At its core, this system uses **LangGraph** to dynamically route user queries to specialized AI agents. Here is how the data flows:

1. **User Query**: The user asks a question via the React frontend.
2. **Hybrid Router**: The backend (FastAPI) receives the query. The LangGraph Router uses a hybrid approach (Keyword Matching + LLM Few-Shot prompting) to intelligently determine the *intent* of the query.
3. **Vector Retrieval**: The system fetches relevant context chunks from a **Pinecone** vector database (embedded using Google's `gemini-embedding-001`).
4. **Specialized Agent Execution**: The router forwards the query and the retrieved context to one of several specialized AI agents.
5. **Evaluation**: The generated response and the source contexts are evaluated mathematically for quality before being sent back to the user.

---

## 🤖 The Specialized Agents

Instead of a generic chatbot, this system relies on a swarm of highly focused agents powered by the `meta-llama/llama-4-scout-17b-16e-instruct` model via Groq:

- **📝 Summarizer**: Condenses long documents into precise, academic paragraphs.
- **✅ MCQ Generator**: Sets university-level multiple-choice questions with options, answers, and explanations.
- **📓 Notes Maker**: Creates highly efficient, structured study notes with side-headings and bullet points.
- **🎓 Exam Prep Agent**: Predicts likely exam questions (categorized into Easy, Medium, and Tough) and creates study plans.
- **💡 Concept Explainer**: Breaks down complex concepts into simple, easy-to-understand terms with analogies.
- **🌐 Search Agent**: If the query requires external, up-to-date knowledge, this agent browses the web using Google Search API.
- **💬 Chat Agent**: Handles standard greetings, conversational pleasantries, and general fallbacks.

> **Note on Prompt Strictness:** All RAG agents have been equipped with strict directives forcing them to rely *exclusively* on the provided context, eliminating AI hallucinations.

---

## 📊 Evaluation System (Ragas)

To ensure the highest quality of AI responses, every interaction is graded using **Ragas**. Two primary metrics are calculated:

1. **Faithfulness**: Measures if the AI hallucinated. A score of `1.0` means the AI's answer was derived 100% from the provided context.
2. **Answer Relevancy**: Measures how directly the AI answered the user's specific query without wandering off-topic.

*The evaluation pipeline uses `meta-llama/llama-4-scout-17b-16e-instruct` as the judge model.*

---

## ⚙️ CI/CD & Docker Setup

This project is fully containerized and equipped with a robust DevOps pipeline.

### Docker
- **Backend**: Containerized via `Dockerfile.backend` (Python 3.10-slim).
- **Frontend**: Containerized via `Dockerfile.frontend` (Node 18-alpine).
- **Docker Compose**: Spin up the entire ecosystem locally by running `docker-compose up`.

### GitHub Actions (CI/CD)
The `.github/workflows/ci-cd.yml` file automates the following on every push to the `main` branch:
1. **Testing**: Runs the `pytest` suite to verify routing logic and API health.
2. **Build & Push**: Builds the frontend and backend Docker images.
3. **Deploy**: Pushes the images directly to Docker Hub. 

*(Requires `DOCKER_USERNAME` and `DOCKER_PASSWORD` repository secrets in GitHub).*

---

## 🚀 Setup & Installation (Local Development)

### 1. Environment Variables
Create a `.env` file in the `backend/` directory with the following keys:
```env
GEMINI_API_KEY=your_key
GOOGLE_API_KEY=your_key
GOOGLE_CSE_ID=your_id
GROQ_API_KEY=your_key
PINECONE_API_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_key
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm start
```

### 4. Running Tests
To verify the system's routing and health:
```bash
cd backend
pytest tests/ -v
```
