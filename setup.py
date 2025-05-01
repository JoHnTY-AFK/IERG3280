import os
import subprocess
import sys
import nltk

def setup_environment():
    """Set up the project environment and verify dependencies."""
    print("Setting up project environment...")
    
    # Create necessary directories
    for folder in ["dataset", "dataset_processed", "index", "src", "tests", "docs", "templates"]:
        os.makedirs(folder, exist_ok=True)
    print("Project directories created.")
    
    # Create virtual environment if not exists
    if not os.path.exists("venv"):
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        print("Virtual environment created.")
    
    # Install dependencies
    try:
        subprocess.check_call([os.path.join("venv", "Scripts" if sys.platform == "win32" else "bin", "pip"), 
                              "install", "-r", "requirements.txt"])
        print("Dependencies installed.")
    except subprocess.CalledProcessError:
        print("Error installing dependencies. Check requirements.txt.")
        sys.exit(1)
    
    # Verify key libraries and download NLTK resources
    try:
        import whoosh
        import flask
        import jinja2
        import pandas
        import nltk
        # Workaround for Werkzeug compatibility
        try:
            from werkzeug.urls import url_quote
        except ImportError:
            print("Warning: Werkzeug version may be incompatible. Ensure werkzeug<3.0.0 is installed.")
        # Download NLTK resources
        for resource in ['punkt', 'stopwords']:
            try:
                nltk.download(resource, quiet=True)
                print(f"NLTK resource '{resource}' downloaded.")
            except Exception as e:
                print(f"Error downloading NLTK resource '{resource}': {e}")
                sys.exit(1)
        print("All dependencies and NLTK resources verified.")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_environment()