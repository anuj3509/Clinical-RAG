#!/usr/bin/env python3
"""
Quick test script to check if all dependencies can be imported
"""

import sys
print(f"Python version: {sys.version}")

try:
    import pandas as pd
    print(f"✓ pandas {pd.__version__} imported successfully")
except ImportError as e:
    print(f"✗ pandas import failed: {e}")

try:
    import chromadb
    print(f"✓ chromadb imported successfully")
except ImportError as e:
    print(f"✗ chromadb import failed: {e}")

try:
    import voyageai
    print(f"✓ voyageai imported successfully")
except ImportError as e:
    print(f"✗ voyageai import failed: {e}")

try:
    from dotenv import load_dotenv
    print(f"✓ python-dotenv imported successfully")
except ImportError as e:
    print(f"✗ python-dotenv import failed: {e}")

try:
    import openai
    print(f"✓ openai imported successfully")
except ImportError as e:
    print(f"✗ openai import failed: {e}")

print("\nTest complete!")
