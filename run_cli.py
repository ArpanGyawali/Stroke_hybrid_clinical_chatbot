#!/usr/bin/env python3
"""CLI launcher script for the medical chatbot."""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Run the CLI version of the medical chatbot."""
    try:
        # Import the CLI main function
        from src.main import main as cli_main
        
        # Set CLI mode
        sys.argv = [sys.argv[0], 'cli'] + sys.argv[1:]
        
        # Run the CLI
        cli_main()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running CLI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()