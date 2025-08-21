# PDF2Table: AI-Powered Data Extraction from Research Papers

PDF2Table is a beginner-friendly, end-to-end AI application that extracts **structured data** from **unstructured PDF research papers**. Built using Python, OpenAI's GPT models, LangChain, ChromaDB, and deployed via Streamlit and Docker, this tool demonstrates how to apply **retrieval-augmented generation (RAG)** techniques to automate the extraction of information like **title, summary, publication year, and authors**.

---

## 🚀 Features

- 📄 Upload research papers in PDF format
- 🤖 Extract structured information using GPT-4 via OpenAI API
- 📚 Retrieve relevant text chunks with LangChain
- 🧠 Embed and search documents using Chroma Vector DB
- 📊 Display structured results in a table using Streamlit
- 📦 Deploy easily using Docker for OS-agnostic execution

---

## 🛠️ Tech Stack

- **Python 3.11**
- **OpenAI API** (GPT-4-turbo or GPT-4o)
- **LangChain** (document loaders, chains, retrievers)
- **ChromaDB** (vector database)
- **Streamlit** (web UI)
- **Docker** (containerization)

---

## 📂 Project Structure

```
PDF2Table/
├── app/
│   ├── streamlit_app.py       # Main Streamlit interface
│   ├── utils.py               # Helper functions for loading, embedding, querying
│   └── ...
├── data/                     # PDF files
├── vectorstore/              # Saved embeddings
├── requirements.txt
├── Dockerfile
├── .env                      # API keys (not shared)
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/PDF2Table.git
cd PDF2Table
```

### 2. Set Up Virtual Environment

```bash
python -m venv myenv
source myenv/bin/activate  # or myenv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your OpenAI API Key

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_key_here
```

---

## 🧪 Run Locally

```bash
streamlit run app/streamlit_app.py
```

---

## 🐳 Run with Docker

### 1. Build Docker Image

```bash
docker build -t pdf2table .
```

### 2. Run Container

```bash
docker run -p 8501:8501 pdf2table
```

Then go to `http://localhost:8501` in your browser.

---

## 🧠 How It Works

1. Load PDF using LangChain's PDF loader.
2. Split content into chunks for better context.
3. Convert chunks into embeddings via OpenAI.
4. Store & retrieve chunks using ChromaDB.
5. Generate structured answers using GPT.
6. Display everything neatly via Streamlit.

---

## 📈 Example Output

| Title                            | Summary                                  | Authors          | Year |
| -------------------------------- | ---------------------------------------- | ---------------- | ---- |
| Estimating Saliva in 5-Year-Olds | A humorous but data-driven estimation... | Dr. ABC, Dr. XYZ | 2006 |

---

## 🙋‍♂️ Why This Project?

Reading and organizing information from research papers is time-consuming. PDF2Table shows how LLMs can **automate knowledge extraction**, making document processing tasks smarter, faster, and more useful.

---

## ✨ Future Ideas

- Add support for multiple PDF uploads
- Extract figures, tables, and references
- Use other embedding models (e.g., HuggingFace)
- Extend to images, invoices, resumes, etc.

---
