#!/usr/bin/env python3
"""
Setup script for Clinical Trials RAG System
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def check_env_file():
    """Check if .env file exists"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("Please create a .env file with your API keys:")
        print("VOYAGE_API_KEY=your_voyage_api_key_here")
        print("OPENAI_API_KEY=your_openai_api_key_here")
        return False
    
    print("✅ .env file found")
    return True

def create_vector_database():
    """Create the vector database"""
    print("🔧 Creating vector database...")
    try:
        subprocess.check_call([sys.executable, "create_vector_db.py"])
        print("✅ Vector database created successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating vector database: {e}")
        return False
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up Clinical Trials RAG System")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check environment file
    if not check_env_file():
        print("\n📝 Next steps:")
        print("1. Create a .env file with your API keys")
        print("2. Run: python setup.py")
        sys.exit(1)
    
    # Create vector database
    if not create_vector_database():
        sys.exit(1)
    
    print("\n🎉 Setup complete!")
    print("\nYou can now use the RAG system:")
    print("• Interactive mode: python rag_system.py interactive")
    print("• Single query: python rag_system.py query 'your question here'")
    print("• Check status: python rag_system.py status")

if __name__ == "__main__":
    main()
