import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import re
from src.utils import log_message

def check_nltk_resources():
    """Check if required NLTK resources are available."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError as e:
        log_message(f"NLTK resource missing: {e}", "error")
        print(f"Error: {e}")
        print("Run: python3 -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\"")
        sys.exit(1)

def preprocess_text(text):
    """Preprocess text by lowercasing, removing headers, stop words, and stemming."""
    try:
        # Initialize NLTK components
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()
        
        # Remove 20 Newsgroups headers and noise
        text = re.sub(r'^From:.*\n?', '', text, flags=re.MULTILINE)  # Remove "From" line
        text = re.sub(r'^Subject:.*\n?', '', text, flags=re.MULTILINE)  # Remove "Subject" line
        text = re.sub(r'^\s*>\s*.*\n?', '', text, flags=re.MULTILINE)  # Remove quoted text
        text = re.sub(r'[\r\n]+', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        
        # Tokenize and normalize
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in string.punctuation]
        tokens = [t for t in tokens if t not in stop_words]
        tokens = [ps.stem(t) for t in tokens]
        
        processed_text = " ".join(tokens)
        # Log the first 100 characters of the processed text for debugging
        log_message(f"Processed text (first 100 chars): {processed_text[:100]}...", "info")
        if not processed_text.strip():
            log_message("Empty text after preprocessing", "warning")
            print("Warning: Empty text after preprocessing")
            return None
        return processed_text
    except Exception as e:
        log_message(f"Preprocessing error: {e}", "error")
        print(f"Error in preprocess_text: {e}")
        return None

def preprocess_file(input_path, output_path):
    """Preprocess a single file and save the result."""
    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        processed_text = preprocess_text(text)
        if processed_text:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            log_message(f"Processed file: {input_path}", "info")
            print(f"Processed: {input_path}")
            return True
        else:
            log_message(f"Skipped file {input_path}: No content after preprocessing", "warning")
            print(f"Skipped: {input_path} (no content)")
            return False
    except Exception as e:
        log_message(f"Error processing {input_path}: {e}", "error")
        print(f"Error processing {input_path}: {e}")
        return False

if __name__ == "__main__":
    # Check NLTK resources
    check_nltk_resources()
    
    input_folder = os.path.join("dataset")
    output_folder = os.path.join("dataset_processed")
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if input folder exists and has files
    if not os.path.exists(input_folder):
        log_message(f"Input folder {input_folder} does not exist", "error")
        print(f"Error: Input folder {input_folder} does not exist")
        sys.exit(1)
    
    files = os.listdir(input_folder)
    if not files:
        log_message(f"No files found in {input_folder}", "error")
        print(f"Error: No files found in {input_folder}")
        sys.exit(1)
    
    log_message(f"Found {len(files)} files in {input_folder}", "info")
    print(f"Found {len(files)} files to process")
    
    categories_seen = set()
    for filename in files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        if preprocess_file(input_path, output_path):
            category = filename.split('_')[0]
            categories_seen.add(category)
    
    # Verify output
    output_files = os.listdir(output_folder)
    log_message(f"Generated {len(output_files)} files in {output_folder}", "info")
    log_message(f"Categories processed: {sorted(list(categories_seen))}", "info")
    print(f"Generated {len(output_files)} files in {output_folder}")
    if len(categories_seen) != 20:
        log_message(f"Warning: Only {len(categories_seen)} categories processed out of expected 20", "warning")
        print(f"Warning: Only {len(categories_seen)} categories processed out of expected 20")