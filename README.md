# Clinical Trials RAG System

A Retrieval-Augmented Generation (RAG) system for querying clinical trials data using ChromaDB vector database, Voyage AI embeddings, and OpenAI for response generation.

## Features

- 🔍 **Vector Search**: Uses Voyage-3 embeddings for semantic search
- 🤖 **AI Responses**: Generates contextual answers using OpenAI GPT
- 💾 **Persistent Storage**: ChromaDB for vector storage
- 🖥️ **CLI Interface**: Interactive and single-query modes
- 📊 **Rich Output**: Formatted tables and panels for better readability

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root with your API keys:

```env
VOYAGE_API_KEY=your_voyage_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Initialize the System

Run the setup script to create the vector database:

```bash
python setup.py
```

Or manually:

```bash
python create_vector_db.py
```

## Usage

### Interactive Mode (Recommended)

Start an interactive session where you can ask multiple questions:

```bash
python rag_system.py interactive
```

### Single Query Mode

Ask a single question and get a response:

```bash
python rag_system.py query "What trials are studying cancer treatments?"
```

### Check Database Status

Verify your vector database is set up correctly:

```bash
python rag_system.py status
```

## Example Queries

Here are some example questions you can ask:

- "What trials are studying Alzheimer's disease?"
- "Show me oncology trials from 2023"
- "What are the primary outcomes for diabetes trials?"
- "Find trials using immunotherapy"
- "What is NCT06155955 about?"
- "Show me trials for spinal cord injury"
- "What trials are studying virtual reality therapy?"

## System Architecture

```
User Query → Voyage AI Embedding → ChromaDB Search → Retrieved Docs → OpenAI GPT → Response
```

1. **Query Processing**: User question is embedded using Voyage-3 model
2. **Vector Search**: ChromaDB finds most similar clinical trial documents
3. **Context Building**: Retrieved documents are formatted as context
4. **Response Generation**: OpenAI GPT generates answer based on context
5. **Result Display**: Formatted response with source information

## Data Structure

The system processes clinical trials data with the following fields:

- **NCT Number**: Unique trial identifier
- **Study Title**: Title of the clinical trial
- **Brief Summary**: Detailed description of the study
- **Conditions**: Medical conditions being studied
- **Primary/Secondary Outcomes**: Study endpoints
- **Start Date**: When the trial began
- **Study Documents**: Links to protocol documents

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `VOYAGE_API_KEY` | Voyage AI API key for embeddings | Yes |
| `OPENAI_API_KEY` | OpenAI API key for response generation | Yes |

### Command Line Options

#### Interactive Mode
```bash
python rag_system.py interactive [OPTIONS]
```

Options:
- `--db-path`: Path to ChromaDB database (default: `./chroma_db`)
- `--model`: OpenAI model to use (default: `gpt-3.5-turbo`)

#### Query Mode
```bash
python rag_system.py query "your question" [OPTIONS]
```

Options:
- `--db-path`: Path to ChromaDB database
- `--model`: OpenAI model to use
- `--n-results`: Number of documents to retrieve (default: 5)

## Files Structure

```
AegisAI-Interview/
├── data/
│   └── download.csv           # Clinical trials data
├── chroma_db/                 # ChromaDB vector database (created)
├── create_vector_db.py        # Script to create vector database
├── rag_system.py             # Main RAG system and CLI
├── setup.py                  # Setup and initialization script
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
└── README.md                # This file
```

## Troubleshooting

### Database Not Found
If you get a "collection not found" error:
```bash
python create_vector_db.py
```

### API Key Issues
Make sure your `.env` file contains valid API keys:
- Get Voyage AI key from: https://www.voyageai.com/
- Get OpenAI key from: https://platform.openai.com/api-keys

### Empty Results
If queries return no results, check:
1. Database was created successfully
2. CSV file is in the correct location (`data/download.csv`)
3. Try broader search terms

## API Costs

- **Voyage AI**: ~$0.10 per 1M tokens for embeddings
- **OpenAI GPT-3.5-turbo**: ~$0.002 per 1K tokens for responses

The initial database creation will embed all clinical trials data. Subsequent queries only embed the user question, making them very cost-effective.

## License

This project is for educational and research purposes.
