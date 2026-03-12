# Complete Setup Guide — Banking RAG Application

This guide provides end-to-end instructions for setting up and running the Banking RAG application, including the core engine, web UI, databases, and the newly integrated Langfuse observability platform.

## Table of Contents

1.  [**Prerequisites**](#prerequisites)
2.  [**Step 1: Unzip Project & Install Dependencies**](#step-1-unzip-project--install-dependencies)
3.  [**Step 2: Set Up Databases & Services**](#step-2-set-up-databases--services)
    *   [Option A: MongoDB (Required)](#option-a-mongodb-required)
    *   [Option B: Langfuse (Optional, Recommended)](#option-b-langfuse-optional-recommended)
4.  [**Step 3: Configure Environment Variables**](#step-3-configure-environment-variables)
5.  [**Step 4: Run the Application**](#step-4-run-the-application)
6.  [**Step 5: First-Time Use (Admin)**](#step-5-first-time-use-admin)
7.  [**Project Structure**](#project-structure)
8.  [**Troubleshooting**](#troubleshooting)

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

| Software | Version | Purpose |
|---|---|---|
| **Python** | 3.10+ | Application runtime |
| **Docker** | Latest | To run self-hosted Langfuse & PostgreSQL |
| **OpenAI API Key** | - | For all LLM calls and embeddings |

---

## Step 1: Unzip Project & Install Dependencies

First, unzip the provided project file and install all required Python packages.

```bash
# 1. Unzip the project archive
unzip banking-rag-app.zip

# 2. Navigate into the project directory
cd banking-rag-app

# 3. Install all Python dependencies
pip install -r requirements.txt
```

This installs key libraries including Streamlit, OpenAI, PyMongo, FAISS, PyMuPDF, and Langfuse.

---

## Step 2: Set Up Databases & Services

The application requires MongoDB for data persistence and can optionally connect to Langfuse for observability.

### Option A: MongoDB (Required)

MongoDB is used to store document metadata, text chunks, and FAQ pairs. Follow the instructions for your operating system.

*   **Windows**: Download and run the [MongoDB Community Server installer](https://www.mongodb.com/try/download/community). Select the "Complete" installation and ensure "Install MongoDB as a Service" is checked. The service will start automatically.
*   **macOS**: Use Homebrew to install and start the service:
    ```bash
    brew tap mongodb/brew
    brew install mongodb-community@7.0
    brew services start mongodb-community@7.0
    ```
*   **Linux (Ubuntu/Debian)**: Use the official repository to install and start the service:
    ```bash
    curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg
    echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] http://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
    sudo apt-get update && sudo apt-get install -y mongodb-org
    sudo systemctl start mongod && sudo systemctl enable mongod
    ```

**Verify MongoDB is running:**

```bash
mongosh --eval "db.runCommand({ ping: 1 })" # Should return { ok: 1 }
```

### Option B: Langfuse (Optional, Recommended)

Langfuse provides detailed tracing and observability for every RAG pipeline execution. You can run it locally via Docker (recommended) or use the cloud version.

**1. Start Self-Hosted Langfuse (via Docker)**

From your `banking-rag-app` directory, run:

```bash
# This starts Langfuse on http://localhost:3100 and a PostgreSQL DB on port 5433
docker compose -f docker-compose-langfuse.yml up -d
```

**2. Create a Langfuse Project & API Keys**

1.  Open **http://localhost:3100** in your browser.
2.  Create an account (the first user becomes the administrator).
3.  Create a new project (e.g., "Banking RAG").
4.  Navigate to **Settings → API Keys → Create New**.
5.  Copy the **Public Key** and **Secret Key**. You will need these in the next step.

---

## Step 3: Configure Environment Variables

The application is configured using a `.env` file. A template is provided.

**1. Create your `.env` file:**

```bash
cp .env.example .env
```

**2. Edit the `.env` file:**

Open the newly created `.env` file in a text editor and fill in the values.

```env
# ============================================
# Banking RAG Application — Environment Variables
# ============================================

# Required: OpenAI API Key (for LLM calls and embeddings)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional: Custom MongoDB connection string (default is fine for local setup)
MONGODB_URI=mongodb://localhost:27017

# ── Langfuse Observability (optional — set to enable tracing) ──
# If you are not using Langfuse, you can leave these blank.
LANGFUSE_HOST=http://localhost:3100
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key-from-langfuse-ui
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key-from-langfuse-ui
```

---

## Step 4: Run the Application

With all dependencies installed and configurations set, you can now run the Streamlit application.

```bash
# Ensure you are in the banking-rag-app directory
streamlit run app.py
```

The application will be available at **http://localhost:8501**.

---

## Step 5: First-Time Use (Admin)

**1. Log In as Admin**

*   **Username**: `admin`
*   **Password**: `admin88$`

**2. Upload Documents**

1.  In the sidebar, find the **"Upload Documents (Admin Only)"** section.
2.  Upload your FAQ PDF files. The system supports `.pdf`, `.txt`, and `.md`.
3.  Wait for the processing to complete. The system will extract text, create chunks, generate embeddings, and build the FAQ index.
4.  Once processed, documents are stored in MongoDB and the FAISS index is saved to disk. You do not need to re-upload them every time you restart the app.

**3. Start Querying**

Log out or use a different browser to log in as a regular user (any username/password will work) and start asking questions.

---

## Project Structure

```
banking-rag-app/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── SETUP.md                        # This setup guide
├── docker-compose-langfuse.yml     # Docker Compose for self-hosted Langfuse
├── config/settings.py              # All application configuration and feature toggles
├── rag_engine/                     # Core 7-Layer RAG pipeline logic
├── persistence/                    # MongoDB and FAISS data persistence
├── governance/                     # AI governance checks (hallucination, bias, etc.)
├── observability/                  # Langfuse integration and tracing logic
├── prompts/                        # Manages all LLM prompts
├── ui/                             # Streamlit UI components
└── data/                           # Stores FAISS index, audit logs, etc.
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| **MongoDB Connection Refused** | Ensure the MongoDB service is running. Use `mongosh --eval "db.runCommand({ ping: 1 })"` to test the connection. Check the service status via `brew services list` (macOS) or `sudo systemctl status mongod` (Linux). |
| **Langfuse Traces Not Appearing** | 1. Verify your `LANGFUSE_*` variables in `.env` are correct. 2. Check that the Langfuse Docker containers are running with `docker ps`. 3. Ensure `LANGFUSE_HOST` matches the port Langfuse is running on (default `http://localhost:3100`). |
| **Slow Document Upload** | In `config/settings.py`, ensure `extraction_config.fast_mode` is `True`. Fast mode skips some of the more intensive (but higher quality) extraction steps. Scanned (image-based) PDFs will always be slower as they require OCR. |
| **Missing FAQ Answers** | If FAQ answers are incomplete or incorrect, the extraction logic may have been updated. Clear the old data from MongoDB (`mongosh banking_rag --eval "db.faq_pairs.drop()"`) and re-upload the document. |
| **Reset Everything** | To start completely fresh, stop the app, delete the `data/faiss_index` directory, and drop all collections in the `banking_rag` MongoDB database. |
