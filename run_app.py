#!/usr/bin/env python3
"""Launcher script for the medical chatbot Streamlit app."""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the main app
if __name__ == "__main__":
    from src.main import main
    main()