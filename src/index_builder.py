import os
import sys
import time
import pickle
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.writing import AsyncWriter
from src.utils import log_message
from src.preprocess import preprocess_text
from nltk.stem import PorterStemmer

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def build_index(input_folder, index_dir):
    """Build an index from preprocessed dataset files with optimized stem mapping."""
    try:
        start_time = time.time()
        log_message(f"Starting index build with input_folder={input_folder}, index_dir={index_dir}", "info")
        
        if not os.path.exists(input_folder):
            log_message(f"Input folder {input_folder} does not exist", "error")
            print(f"Error: Input folder {input_folder} does not exist")
            return
        
        # Define schema for the index
        schema = Schema(
            doc_id=ID(stored=True),
            content=TEXT(stored=True),
            original_content=TEXT(stored=True),
            category=ID(stored=True)
        )
        
        # Create or open the index directory
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
            ix = create_in(index_dir, schema)
        else:
            # Clear the existing index to avoid corruption
            for file in os.listdir(index_dir):
                file_path = os.path.join(index_dir, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            ix = create_in(index_dir, schema)
        
        # Initialize stemmer and collect terms for mapping
        ps = PorterStemmer()
        stem_cache = {}  # Cache for stemmed terms to avoid redundant stemming
        stem_mapping = {}  # Map stemmed terms to unstemmed terms
        writer = AsyncWriter(ix)
        processed_files = 0
        categories = set()
        total_files = sum(1 for f in os.listdir(input_folder) if f.endswith('.txt'))
        
        # Single pass: Index documents and build stem mapping
        for filename in os.listdir(input_folder):
            if not filename.endswith('.txt'):
                continue
            filepath = os.path.join(input_folder, filename)
            raw_filepath = os.path.join("dataset", filename)
            
            try:
                # Read preprocessed file with error handling
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                except UnicodeDecodeError:
                    log_message(f"UTF-8 decoding failed for {filepath}, falling back to latin-1", "warning")
                    with open(filepath, 'r', encoding='latin-1') as f:
                        content = f.read().strip()
                
                if not content:
                    log_message(f"Empty processed content in {filename}", "warning")
                    continue
                
                # Read raw file with error handling
                if os.path.exists(raw_filepath):
                    try:
                        with open(raw_filepath, 'r', encoding='utf-8') as f:
                            original_content = f.read().strip()
                    except UnicodeDecodeError:
                        log_message(f"UTF-8 decoding failed for {raw_filepath}, falling back to latin-1", "warning")
                        with open(raw_filepath, 'r', encoding='latin-1') as f:
                            original_content = f.read().strip()
                else:
                    log_message(f"Raw file {raw_filepath} not found, using processed content", "warning")
                    original_content = content
                
                category = filename.split('_')[0]
                categories.add(category)
                
                # Build stem mapping in a single pass
                tokens = content.split()
                original_tokens = original_content.split()
                # Process pairs of stemmed and unstemmed tokens
                for token, orig_token in zip(tokens, original_tokens):
                    # Use cached stemming result if available
                    token_lower = token.lower()
                    if token_lower in stem_cache:
                        stemmed = stem_cache[token_lower]
                    else:
                        stemmed = ps.stem(token_lower)
                        stem_cache[token_lower] = stemmed
                    
                    if stemmed not in stem_mapping:
                        stem_mapping[stemmed] = set()
                    stem_mapping[stemmed].add(orig_token.lower())
                
                writer.add_document(
                    doc_id=filename,
                    content=content,
                    original_content=original_content,
                    category=category
                )
                processed_files += 1
                if processed_files % 100 == 0:
                    log_message(f"Processed {processed_files}/{total_files} files", "info")
                    print(f"Processed {processed_files}/{total_files} files")
            
            except Exception as e:
                log_message(f"Error indexing {filename}: {str(e)}", "error")
                continue
        
        log_message("Committing indexed documents...", "info")
        print("Committing indexed documents...")
        writer.commit()
        
        # Save stem mapping
        mapping_file = os.path.join(index_dir, "stem_mapping.pkl")
        with open(mapping_file, 'wb') as f:
            pickle.dump(stem_mapping, f)
        log_message(f"Saved stem mapping to {mapping_file}", "info")
        
        elapsed_time = time.time() - start_time
        log_message(f"Index build completed in {elapsed_time:.2f} seconds. Indexed {processed_files} files.", "info")
        log_message(f"Categories indexed: {sorted(list(categories))}", "info")
        print(f"Index build completed in {elapsed_time:.2f} seconds. Indexed {processed_files} files.")
    
    except KeyboardInterrupt:
        log_message("Index build interrupted by user", "warning")
        print("Index build interrupted. Committing progress...")
        writer.commit()
        # Save partial stem mapping if it exists
        if 'stem_mapping' in locals():
            mapping_file = os.path.join(index_dir, "stem_mapping.pkl")
            with open(mapping_file, 'wb') as f:
                pickle.dump(stem_mapping, f)
            log_message(f"Saved partial stem mapping to {mapping_file}", "info")
        elapsed_time = time.time() - start_time
        log_message(f"Partial index build stopped after {elapsed_time:.2f} seconds. Indexed {processed_files} files.", "info")
        print(f"Partial index build stopped after {elapsed_time:.2f} seconds. Indexed {processed_files} files.")
    except Exception as e:
        log_message(f"Index build failed: {str(e)}", "error")
        print(f"Error during index build: {str(e)}")

if __name__ == "__main__":
    build_index("dataset_processed", "index")