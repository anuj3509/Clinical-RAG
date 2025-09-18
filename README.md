# Clinical Trials RAG System

A Retrieval-Augmented Generation (RAG) system for querying clinical trials data using ChromaDB vector database, Voyage AI embeddings, and OpenAI for response generation.

## Features

- 🔍 **Advanced Search**: Hybrid search combining exact NCT number matching with semantic vector search
- 🤖 **AI Responses**: Generates contextual answers using OpenAI GPT models
- 💾 **Persistent Storage**: ChromaDB vector database with Voyage-3 embeddings
- 🖥️ **CLI Interface**: Interactive and single-query modes with rich formatting
- 📊 **Rich Output**: Formatted tables, panels, and markdown for better readability
- 🎯 **Smart Retrieval**: Prioritizes exact trial matches while providing related studies
- 🔧 **Configurable**: Customizable models, database paths, and retrieval parameters
- 📈 **Comprehensive Coverage**: 20+ clinical trials across multiple medical specialties

## Quick Start

```bash
# 1. Clone and navigate to the project
git clone <repository_url>
cd AegisAI-Interview

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
echo "VOYAGE_API_KEY=your_voyage_api_key_here" > .env
echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env

# 4. Initialize the database
python setup.py

# 5. Start querying!
python rag_system.py interactive
```

## Detailed Setup

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

**Getting API Keys:**
- **Voyage AI**: Sign up at [voyageai.com](https://www.voyageai.com/) 
- **OpenAI**: Get your key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 3. Initialize the System

Run the setup script to create the vector database:

```bash
python setup.py
```

Or manually:

```bash
python create_vector_db.py
```

This will:
- Load clinical trials data from `data/download.csv`
- Generate embeddings using Voyage AI
- Create a ChromaDB vector database in `./chroma_db/`

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

Here are example questions you can ask:

### 🔬 **Disease-Specific Queries**
- "What trials are studying spinal cord injury?"
- "Find trials investigating hemophilia treatments"
- "Show me studies on multiple myeloma"

### 🏥 **Treatment & Intervention Queries**
- "What trials are using virtual reality therapy?"
- "Show me immunotherapy trials"
- "Find trials investigating yoga interventions"

### 📊 **Specific Trial Information**
- "What is NCT06155955 about?"
- "What are the primary outcomes for NCT06154122?"
- "Show me details about the VR spinal cord injury trial"

### 🎯 **Outcomes & Research Focus**
- "Show me trials measuring quality of life"
- "Find studies with pain reduction endpoints"
- "What randomized controlled trials are available?"

## Sample Clinical Trials in Database

The database contains diverse clinical trials across multiple medical specialties:

### 🩸 **Hematology**
- **NCT06155955**: Low dose Emicizumab vs Factor VIII in severe hemophilia A
- **NCT06158269**: DVRd treatment for newly diagnosed double-hit multiple myeloma

### 🧠 **Neurology & Rehabilitation**  
- **NCT06154122**: Virtual reality upper limb therapy for spinal cord injury patients
- **NCT06153992**: Intelligent multi-joint isokinetic training system for stroke rehabilitation
- **NCT06156735**: Music therapy effects on cognition in neurorehabilitation
- **NCT06159946**: Smart faucet technology for spinal cord injury patients

### 🦴 **Orthopedics & Sports Medicine**
- **NCT06150378**: Platelet-rich plasma vs corticosteroids for rotator cuff tendinopathy
- **NCT06152900**: Electronic muscle stimulation for circumferential reduction and toning
- **NCT06147739**: Temporary anchorage devices for anterior open bite treatment

### 🫀 **Cardiology & Internal Medicine**
- **NCT06155240**: Yoga intervention for hypertension in African American patients
- **NCT06155006**: Hepatitis C screening and treatment linkage in Colombian hospitals

### 🏥 **Surgery & Procedures**
- **NCT06152679**: Patient experience with day-case endourology procedures
- **NCT06152666**: Healthcare staff perspectives on day-case endourology barriers
- **NCT06152952**: Rhomboid flap vs deep suturing for recurrent pilonidal sinus

### 🧬 **Oncology & Immunotherapy**
- **NCT06152367**: Autologous dendritic cell immunization for melanoma patients
- **NCT06149832**: Umbilical cord mesenchymal stem cells for oral graft-vs-host disease

### 🤱 **Obstetrics & Gynecology**
- **NCT06157684**: Exercise timing effects on infant birth weight in gestational diabetes
- **NCT06145295**: Online support groups for ovarian cancer patients

### 📊 **Research & Natural History Studies**
- **NCT06151600**: Natural history study of Charcot-Marie-Tooth disease type 4J
- **NCT06145646**: Temperature effects on muscle ultrasound characteristics

## Advanced Usage Tips

### 🎯 **Query Optimization**
- **Be Specific**: Use medical terminology and specific conditions
- **Use NCT Numbers**: For exact trial information, include NCT numbers
- **Combine Keywords**: Mix condition + treatment for better results
- **Ask Follow-ups**: Build on previous queries for deeper insights

### 💡 **Pro Tips**
- **Exact Matches**: NCT numbers get priority in search results
- **Related Studies**: System finds similar trials even for specific queries  
- **Multiple Formats**: Ask questions naturally - the AI understands context
- **Outcome Focus**: Query specific endpoints like "pain reduction" or "quality of life"

### 📋 **Query Examples by Use Case**

**For Researchers:**
- "What trials measure [specific outcome] in [condition]?"
- "Show me [intervention] studies with [duration] follow-up"
- "Find trials using [assessment tool] for [population]"

**For Clinicians:**
- "What treatments are being studied for [patient condition]?"
- "Show me recent trials for [specific therapy]"
- "What are the inclusion criteria for [disease] trials?"

**For Patients/Advocates:**
- "What trials are available for [my condition]?"
- "Show me studies in [location/timeframe]"
- "What are the side effects being monitored in [treatment] trials?"

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

### 🔍 **Database Not Found**
```bash
Error: collection 'clinical_trials' not found
```
**Solution:**
```bash
python create_vector_db.py
```

### 🔑 **API Key Issues**
```bash
ValueError: VOYAGE_API_KEY not found in environment variables
```
**Solutions:**
1. Check your `.env` file exists in the project root
2. Verify API keys are correctly formatted:
   - Voyage AI keys start with `vo-` or `voy-`
   - OpenAI keys start with `sk-`
3. Restart your terminal after creating `.env`
4. Test API keys:
```bash
python test_setup.py
```

### 📊 **Empty Results**
If queries return "no relevant clinical trials found":

**Check database status:**
```bash
python rag_system.py status
```

**Common fixes:**
1. Ensure database was created: `python create_vector_db.py`
2. Verify CSV file exists: `data/download.csv`
3. Try broader search terms
4. Check specific NCT numbers: `"What is NCT06155955 about?"`

### 🐛 **Import Errors**
```bash
ModuleNotFoundError: No module named 'voyageai'
```
**Solution:**
```bash
pip install -r requirements.txt
```

### 💾 **Database Corruption**
If you encounter ChromaDB errors:
```bash
rm -rf ./chroma_db
python create_vector_db.py
```

### 🌐 **Network Issues**
If API calls fail:
1. Check internet connection
2. Verify API key quotas/limits
3. Try different OpenAI model: `--model gpt-4`

### 🔧 **Performance Issues**
- Use `gpt-3.5-turbo` for faster responses
- Reduce `--n-results` parameter for quicker searches
- Check available system memory

## API Costs

- **Voyage AI**: ~$0.10 per 1M tokens for embeddings
- **OpenAI GPT-3.5-turbo**: ~$0.002 per 1K tokens for responses

The initial database creation will embed all clinical trials data. Subsequent queries only embed the user question, making them very cost-effective.

## License

This project is for educational and research purposes.
