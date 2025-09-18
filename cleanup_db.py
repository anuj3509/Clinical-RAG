#!/usr/bin/env python3
"""
Clean up ChromaDB to prevent duplicate ID warnings
"""

import os
import shutil
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress ChromaDB warnings
chromadb_logger = logging.getLogger('chromadb')
chromadb_logger.setLevel(logging.ERROR)

def cleanup_database(db_path: str = "./chroma_db", backup: bool = True):
    """
    Clean up the ChromaDB database by removing and recreating it
    
    Args:
        db_path: Path to the ChromaDB database
        backup: Whether to create a backup before cleaning
    """
    try:
        if not os.path.exists(db_path):
            logger.info(f"Database path {db_path} does not exist. Nothing to clean.")
            return
        
        # Create backup if requested
        if backup:
            backup_path = f"{db_path}_backup_{int(__import__('time').time())}"
            logger.info(f"Creating backup at {backup_path}")
            shutil.copytree(db_path, backup_path)
            logger.info(f"Backup created successfully")
        
        # Remove the database
        logger.info(f"Removing database at {db_path}")
        shutil.rmtree(db_path)
        logger.info(f"Database removed successfully")
        
        # Recreate the database
        logger.info("Recreating database...")
        from create_vector_db import main
        main()
        
        logger.info("Database cleanup and recreation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during database cleanup: {e}")
        raise

def main():
    """Main cleanup function"""
    print("🧹 ChromaDB Cleanup Tool")
    print("=" * 40)
    
    response = input("This will remove and recreate your vector database. Continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Cleanup cancelled.")
        return
    
    backup_response = input("Create backup before cleanup? (Y/n): ")
    create_backup = backup_response.lower() not in ['n', 'no']
    
    try:
        cleanup_database(backup=create_backup)
        print("\n✅ Cleanup completed successfully!")
        print("Your database has been recreated and should no longer show duplicate ID warnings.")
        
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
        print("You may need to manually remove the chroma_db directory and run create_vector_db.py")

if __name__ == "__main__":
    main()
