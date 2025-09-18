#!/usr/bin/env python3
"""
RAG System for Clinical Trials Database
Provides retrieval-augmented generation using ChromaDB and OpenAI/Voyage AI
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Disable ChromaDB telemetry BEFORE importing chromadb
os.environ['CHROMA_TELEMETRY_DISABLED'] = 'true'

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown

import openai
import voyageai
import chromadb
from chromadb.config import Settings

# Custom logging filter to suppress ChromaDB telemetry errors
class ChromaDBTelemetryFilter(logging.Filter):
    def filter(self, record):
        # Filter out ChromaDB telemetry error messages
        if "Failed to send telemetry event" in record.getMessage():
            return False
        if "capture() takes 1 positional argument but 3 were given" in record.getMessage():
            return False
        return True

# Set up logging with custom configuration to suppress ChromaDB noise
class QuietFormatter(logging.Formatter):
    def format(self, record):
        # Skip ChromaDB telemetry errors entirely
        if "Failed to send telemetry event" in record.getMessage():
            return ""
        if "capture() takes 1 positional argument but 3 were given" in record.getMessage():
            return ""
        return super().format(record)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply custom formatter to suppress telemetry noise
for handler in logging.root.handlers:
    handler.setFormatter(QuietFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    handler.addFilter(ChromaDBTelemetryFilter())

# Suppress ChromaDB at logger level too
chromadb_logger = logging.getLogger('chromadb')
chromadb_logger.setLevel(logging.CRITICAL)  # Only show critical errors

# Initialize console for rich output
console = Console()

@dataclass
class RetrievedDocument:
    """Represents a retrieved document with metadata"""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    nct_number: str
    study_title: str

class ClinicalTrialsRAG:
    """RAG System for Clinical Trials Database"""
    
    def __init__(self, 
                 db_path: str = "./chroma_db",
                 collection_name: str = "clinical_trials",
                 openai_model: str = "gpt-3.5-turbo",
                 max_context_length: int = 4000):
        """
        Initialize the RAG system
        
        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of the collection
            openai_model: OpenAI model to use for generation
            max_context_length: Maximum context length for generation
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.openai_model = openai_model
        self.max_context_length = max_context_length
        
        # Initialize API clients
        self._init_clients()
        
        # Load ChromaDB collection
        self._load_collection()
    
    def _init_clients(self):
        """Initialize API clients"""
        # OpenAI client
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        openai.api_key = openai_key
        self.openai_client = openai
        
        # Voyage AI client
        voyage_key = os.getenv('VOYAGE_API_KEY')
        if not voyage_key:
            raise ValueError("VOYAGE_API_KEY not found in environment variables")
        
        self.voyage_client = voyageai.Client(api_key=voyage_key)
        
        logger.info("API clients initialized successfully")
    
    def _load_collection(self):
        """Load ChromaDB collection"""
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            
            count = self.collection.count()
            logger.info(f"Loaded collection '{self.collection_name}' with {count} documents")
            
            if count == 0:
                console.print("[red]Warning: Collection is empty. Please run create_vector_db.py first.[/red]")
        
        except Exception as e:
            console.print(f"[red]Error loading collection: {e}[/red]")
            console.print("[yellow]Please run create_vector_db.py to create the vector database first.[/yellow]")
            sys.exit(1)
    
    def retrieve(self, query: str, n_results: int = 5) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents using hybrid search (exact match + vector similarity)
        
        Args:
            query: Search query
            n_results: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        try:
            # Check if query contains an NCT number
            import re
            nct_pattern = r'NCT\d{8}'
            nct_matches = re.findall(nct_pattern, query.upper())
            
            if nct_matches:
                # First, try exact NCT number match
                nct_number = nct_matches[0]
                
                # Get all documents and check for exact NCT match
                all_results = self.collection.get(
                    include=['documents', 'metadatas']
                )
                
                exact_matches = []
                for i, metadata in enumerate(all_results['metadatas']):
                    if metadata.get('nct_number') == nct_number:
                        doc = RetrievedDocument(
                            content=all_results['documents'][i],
                            metadata=metadata,
                            similarity_score=1.0,  # Perfect match
                            nct_number=metadata.get('nct_number', 'Unknown'),
                            study_title=metadata.get('study_title', 'Unknown')
                        )
                        exact_matches.append(doc)
                
                if exact_matches:
                    # If we found exact matches, prioritize them but also get similar ones
                    remaining_results = n_results - len(exact_matches)
                    if remaining_results > 0:
                        # Get vector similarity results
                        query_embedding = self.voyage_client.embed(
                            texts=[query],
                            model="voyage-3",
                            input_type="query"
                        ).embeddings[0]
                        
                        vector_results = self.collection.query(
                            query_embeddings=[query_embedding],
                            n_results=remaining_results + len(exact_matches),
                            include=['documents', 'metadatas', 'distances']
                        )
                        
                        # Add non-duplicate vector results
                        for i, (doc, metadata, distance) in enumerate(zip(
                            vector_results['documents'][0],
                            vector_results['metadatas'][0],
                            vector_results['distances'][0]
                        )):
                            vector_nct = metadata.get('nct_number', '')
                            if vector_nct != nct_number:  # Avoid duplicates
                                similarity_score = 1 - distance
                                retrieved_doc = RetrievedDocument(
                                    content=doc,
                                    metadata=metadata,
                                    similarity_score=similarity_score,
                                    nct_number=vector_nct,
                                    study_title=metadata.get('study_title', 'Unknown')
                                )
                                exact_matches.append(retrieved_doc)
                                if len(exact_matches) >= n_results:
                                    break
                    
                    logger.info(f"Retrieved {len(exact_matches)} documents (hybrid search)")
                    return exact_matches[:n_results]
            
            # Fallback to regular vector similarity search
            query_embedding = self.voyage_client.embed(
                texts=[query],
                model="voyage-3",
                input_type="query"
            ).embeddings[0]
            
            # Search the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert to RetrievedDocument objects
            retrieved_docs = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                similarity_score = 1 - distance  # Convert distance to similarity
                
                retrieved_doc = RetrievedDocument(
                    content=doc,
                    metadata=metadata,
                    similarity_score=similarity_score,
                    nct_number=metadata.get('nct_number', 'Unknown'),
                    study_title=metadata.get('study_title', 'Unknown')
                )
                retrieved_docs.append(retrieved_doc)
            
            return retrieved_docs
        
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def generate_response(self, query: str, retrieved_docs: List[RetrievedDocument]) -> str:
        """
        Generate response using retrieved documents
        
        Args:
            query: Original user query
            retrieved_docs: Retrieved documents
            
        Returns:
            Generated response
        """
        if not retrieved_docs:
            return "I couldn't find any relevant clinical trials for your query. Please try rephrasing your question."
        
        # Build context from retrieved documents
        context_parts = []
        total_length = 0
        
        for doc in retrieved_docs:
            doc_text = f"Study: {doc.nct_number} - {doc.study_title}\nContent: {doc.content}\n"
            
            if total_length + len(doc_text) > self.max_context_length:
                break
            
            context_parts.append(doc_text)
            total_length += len(doc_text)
        
        context = "\n---\n".join(context_parts)
        
        
        # Create prompt
        system_prompt = """You are an expert medical research assistant specializing in clinical trials. 
        You help users find and understand information about clinical trials based on the provided context.
        
        Guidelines:
        - Provide accurate, helpful information based only on the context provided
        - If asked about specific trials, mention their NCT numbers
        - Be concise but comprehensive
        - If the context doesn't contain enough information to answer fully, say so
        - Focus on the most relevant trials for the user's query
        """
        
        user_prompt = f"""Based on the following clinical trials information, please answer this question: {query}

        Clinical Trials Context:
        {context}
        
        Please provide a helpful response based on the information above."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while generating the response: {e}"
    
    def query(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve + generate
        
        Args:
            question: User question
            n_results: Number of documents to retrieve
            
        Returns:
            Dictionary with response and metadata
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question, n_results)
        
        # Generate response
        response = self.generate_response(question, retrieved_docs)
        
        return {
            'question': question,
            'response': response,
            'retrieved_docs': retrieved_docs,
            'n_retrieved': len(retrieved_docs)
        }

class RAGCLIInterface:
    """CLI Interface for the RAG system"""
    
    def __init__(self, rag_system: ClinicalTrialsRAG):
        self.rag = rag_system
    
    def display_welcome(self):
        """Display welcome message"""
        welcome_text = """
        🏥 Clinical Trials RAG System
        
        Ask questions about clinical trials and get AI-powered responses
        based on the clinical trials database.
        
        Example queries:
        • "What trials are studying cancer treatments?"
        • "Show me trials for diabetes"
        • "What are the primary outcomes for NCT06155955?"
        • "Find trials starting in 2023"
        """
        
        console.print(Panel(welcome_text, title="Welcome", border_style="blue"))
    
    def display_results(self, result: Dict[str, Any]):
        """Display query results"""
        # Display the AI response
        console.print(Panel(
            Markdown(result['response']), 
            title="🤖 AI Response", 
            border_style="green"
        ))
        
        # Display retrieved documents
        if result['retrieved_docs']:
            console.print(f"\n📚 Retrieved {result['n_retrieved']} relevant trials:")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("NCT Number", style="cyan", width=12)
            table.add_column("Title", style="white", width=40)
            table.add_column("Similarity", style="green", width=10)
            
            for doc in result['retrieved_docs'][:3]:  # Show top 3
                similarity_pct = f"{doc.similarity_score:.1%}"
                title_truncated = doc.study_title[:37] + "..." if len(doc.study_title) > 40 else doc.study_title
                
                table.add_row(
                    doc.nct_number,
                    title_truncated,
                    similarity_pct
                )
            
            console.print(table)
    
    def run_interactive(self):
        """Run interactive CLI session"""
        self.display_welcome()
        
        console.print("\n[yellow]Type 'quit' or 'exit' to end the session[/yellow]")
        console.print("[yellow]Type 'help' for more information[/yellow]\n")
        
        while True:
            try:
                # Get user input
                query = console.input("[bold blue]❓ Your question: [/bold blue]")
                
                if not query.strip():
                    continue
                
                # Handle special commands
                if query.lower() in ['quit', 'exit', 'q']:
                    console.print("[green]Goodbye! 👋[/green]")
                    break
                
                if query.lower() in ['help', 'h']:
                    self.display_help()
                    continue
                
                # Process query
                console.print("[yellow]🔍 Searching and generating response...[/yellow]")
                
                result = self.rag.query(query)
                self.display_results(result)
                
                console.print()  # Add spacing
                
            except KeyboardInterrupt:
                console.print("\n[green]Goodbye! 👋[/green]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    def display_help(self):
        """Display help information"""
        help_text = """
        🆘 Help - Clinical Trials RAG System
        
        Commands:
        • Type any question about clinical trials
        • 'help' or 'h' - Show this help message
        • 'quit', 'exit', or 'q' - Exit the program
        
        Example queries:
        • "What trials are studying Alzheimer's disease?"
        • "Show me oncology trials from 2023"
        • "What are the inclusion criteria for diabetes trials?"
        • "Find trials using immunotherapy"
        • "What is the primary outcome for NCT06155955?"
        
        Tips:
        • Be specific in your questions for better results
        • You can ask about conditions, treatments, outcomes, dates, etc.
        • The system will show you the most relevant trials and provide an AI summary
        """
        
        console.print(Panel(help_text, title="Help", border_style="yellow"))

# CLI Commands
@click.group()
def cli():
    """Clinical Trials RAG System - Query clinical trials with AI assistance"""
    pass

@cli.command()
@click.option('--db-path', default='./chroma_db', help='Path to ChromaDB database')
@click.option('--model', default='gpt-3.5-turbo', help='OpenAI model to use')
def interactive(db_path, model):
    """Start interactive CLI session"""
    try:
        rag_system = ClinicalTrialsRAG(db_path=db_path, openai_model=model)
        cli_interface = RAGCLIInterface(rag_system)
        cli_interface.run_interactive()
    except Exception as e:
        console.print(f"[red]Error initializing RAG system: {e}[/red]")
        sys.exit(1)

@cli.command()
@click.argument('question')
@click.option('--db-path', default='./chroma_db', help='Path to ChromaDB database')
@click.option('--model', default='gpt-3.5-turbo', help='OpenAI model to use')
@click.option('--n-results', default=5, help='Number of documents to retrieve')
def query(question, db_path, model, n_results):
    """Ask a single question and get response"""
    try:
        rag_system = ClinicalTrialsRAG(db_path=db_path, openai_model=model)
        result = rag_system.query(question, n_results)
        
        cli_interface = RAGCLIInterface(rag_system)
        cli_interface.display_results(result)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

@cli.command()
@click.option('--db-path', default='./chroma_db', help='Path to ChromaDB database')
def status(db_path):
    """Check the status of the vector database"""
    try:
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_collection(name="clinical_trials")
        count = collection.count()
        
        console.print(f"[green]✅ Database Status[/green]")
        console.print(f"Database Path: {db_path}")
        console.print(f"Collection: clinical_trials")
        console.print(f"Documents: {count}")
        
        if count == 0:
            console.print("[yellow]⚠️  Database is empty. Run 'python create_vector_db.py' to populate it.[/yellow]")
        
    except Exception as e:
        console.print(f"[red]❌ Database not found or error: {e}[/red]")
        console.print("[yellow]Run 'python create_vector_db.py' to create the database first.[/yellow]")

if __name__ == "__main__":
    cli()
