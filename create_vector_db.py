#!/usr/bin/env python3
"""
Script to create a ChromaDB vector database with Voyage AI embeddings
for clinical trials data from CSV file.
"""

import os
import pandas as pd
import chromadb
from chromadb.config import Settings
import voyageai
from dotenv import load_dotenv
from typing import List, Dict, Any
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ClinicalTrialsVectorDB:
    def __init__(self, voyage_api_key: str = None, db_path: str = "./chroma_db"):
        """
        Initialize the Clinical Trials Vector Database
        
        Args:
            voyage_api_key: Voyage AI API key
            db_path: Path to store ChromaDB data
        """
        self.voyage_api_key = voyage_api_key or os.getenv('VOYAGE_API_KEY')
        if not self.voyage_api_key:
            raise ValueError("Voyage AI API key is required. Set VOYAGE_API_KEY environment variable.")
        
        self.db_path = db_path
        self.voyage_client = voyageai.Client(api_key=self.voyage_api_key)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Create or get collection
        self.collection_name = "clinical_trials"
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Clinical trials data with Voyage AI embeddings"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load and preprocess the CSV data
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Processed pandas DataFrame
        """
        logger.info(f"Loading CSV data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Fill NaN values with empty strings
        df = df.fillna('')
        
        logger.info(f"Loaded {len(df)} records from CSV")
        return df
    
    def create_document_text(self, row: pd.Series) -> str:
        """
        Create a comprehensive text representation of each clinical trial
        
        Args:
            row: A row from the DataFrame
            
        Returns:
            Formatted text string for embedding
        """
        text_parts = []
        
        # Add title and NCT number
        if row['NCT Number']:
            text_parts.append(f"NCT Number: {row['NCT Number']}")
        
        if row['Study Title']:
            text_parts.append(f"Title: {row['Study Title']}")
        
        # Add brief summary
        if row['Brief Summary']:
            text_parts.append(f"Summary: {row['Brief Summary']}")
        
        # Add conditions
        if row['Conditions']:
            text_parts.append(f"Conditions: {row['Conditions']}")
        
        # Add outcome measures
        if row['Primary Outcome Measures']:
            text_parts.append(f"Primary Outcomes: {row['Primary Outcome Measures']}")
        
        if row['Secondary Outcome Measures']:
            text_parts.append(f"Secondary Outcomes: {row['Secondary Outcome Measures']}")
        
        # Add start date
        if row['Start Date']:
            text_parts.append(f"Start Date: {row['Start Date']}")
        
        return " | ".join(text_parts)
    
    def get_voyage_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Get embeddings from Voyage AI in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            try:
                # Use voyage-3 model for high-quality embeddings
                response = self.voyage_client.embed(
                    texts=batch,
                    model="voyage-3",
                    input_type="document"
                )
                all_embeddings.extend(response.embeddings)
            except Exception as e:
                logger.error(f"Error getting embeddings for batch {i//batch_size + 1}: {e}")
                # Add zero embeddings as fallback
                embedding_dim = 1024  # voyage-3 dimension
                all_embeddings.extend([[0.0] * embedding_dim for _ in batch])
        
        return all_embeddings
    
    def add_data_to_collection(self, df: pd.DataFrame):
        """
        Process CSV data and add to ChromaDB collection
        
        Args:
            df: DataFrame containing the clinical trials data
        """
        logger.info("Processing data for embedding...")
        
        # Create document texts
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            # Create document text for embedding
            doc_text = self.create_document_text(row)
            documents.append(doc_text)
            
            # Create metadata
            metadata = {
                'nct_number': str(row['NCT Number']),
                'study_title': str(row['Study Title']),
                'conditions': str(row['Conditions']),
                'start_date': str(row['Start Date']),
                'has_primary_outcomes': bool(row['Primary Outcome Measures']),
                'has_secondary_outcomes': bool(row['Secondary Outcome Measures']),
                'has_study_documents': bool(row['Study Documents'])
            }
            metadatas.append(metadata)
            
            # Create unique ID
            ids.append(f"trial_{idx}")
        
        # Get embeddings from Voyage AI
        logger.info("Getting embeddings from Voyage AI...")
        embeddings = self.get_voyage_embeddings(documents)
        
        # Add to ChromaDB collection
        logger.info("Adding data to ChromaDB collection...")
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully added {len(documents)} documents to the collection")
    
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search the vector database
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Search results
        """
        # Get query embedding
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
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection
        
        Returns:
            Collection information
        """
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'document_count': count,
            'db_path': self.db_path
        }

def main():
    """
    Main function to create the vector database
    """
    # Check if API key is available
    api_key = os.getenv('VOYAGE_API_KEY')
    if not api_key:
        logger.error("VOYAGE_API_KEY not found in environment variables.")
        logger.info("Please set your Voyage AI API key:")
        logger.info("1. Copy env_example.txt to .env")
        logger.info("2. Add your Voyage AI API key to the .env file")
        logger.info("3. Or export VOYAGE_API_KEY=your_key_here")
        return
    
    try:
        # Initialize the vector database
        vector_db = ClinicalTrialsVectorDB()
        
        # Load CSV data
        csv_path = "data/download.csv"
        df = vector_db.load_csv_data(csv_path)
        
        # Add data to collection
        vector_db.add_data_to_collection(df)
        
        # Print collection info
        info = vector_db.get_collection_info()
        logger.info(f"Vector database created successfully!")
        logger.info(f"Collection: {info['collection_name']}")
        logger.info(f"Documents: {info['document_count']}")
        logger.info(f"Database path: {info['db_path']}")
        
        # Test with a sample query
        logger.info("\nTesting with sample query...")
        results = vector_db.search("cancer treatment trials", n_results=3)
        
        logger.info("Sample search results:")
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            logger.info(f"\nResult {i+1} (similarity: {1-distance:.3f}):")
            logger.info(f"NCT: {metadata['nct_number']}")
            logger.info(f"Title: {metadata['study_title'][:100]}...")
            
    except Exception as e:
        logger.error(f"Error creating vector database: {e}")
        raise

if __name__ == "__main__":
    main()
