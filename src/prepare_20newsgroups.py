import os
import shutil
import sys
import random
import argparse
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils import log_message

def prepare_20newsgroups(source_dir, target_dir, max_docs=None):
    """Copy 20 Newsgroups files to dataset/ and rename to .txt."""
    try:
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Collect all files grouped by category
        category_files = {}
        total_files = 0
        for group in os.listdir(source_dir):
            group_path = os.path.join(source_dir, group)
            if os.path.isdir(group_path):
                files = [
                    (group, filename)
                    for filename in os.listdir(group_path)
                    if os.path.isfile(os.path.join(group_path, filename))
                    and os.path.getsize(os.path.join(group_path, filename)) >= 100
                ]
                category_files[group] = files
                total_files += len(files)
                log_message(f"Found {len(files)} files in category {group}", "info")
        
        if total_files == 0:
            log_message("No valid files found in source directory", "error")
            print("Error: No valid files found in source directory")
            sys.exit(1)
        
        log_message(f"Total files available: {total_files} across {len(category_files)} categories", "info")
        print(f"Total files available: {total_files} across {len(category_files)} categories")
        
        # Determine how many files to select
        max_docs = total_files if max_docs is None else min(max_docs, total_files)
        log_message(f"Target number of documents to process: {max_docs}", "info")
        
        # Calculate proportional sampling
        selected_files = []
        files_per_category = {}
        for category, files in category_files.items():
            # Proportional number of files for this category
            proportion = len(files) / total_files
            target_num = int(proportion * max_docs)
            # Ensure at least one file per category if possible
            target_num = max(1, target_num) if len(files) > 0 else 0
            files_per_category[category] = target_num
        
        # Adjust for rounding errors to match max_docs exactly
        total_selected = sum(files_per_category.values())
        while total_selected != max_docs:
            # Find a category to adjust
            adjustment = 1 if total_selected < max_docs else -1
            # Adjust the category with the most files remaining
            adjustable_categories = [
                cat for cat, count in files_per_category.items()
                if (adjustment > 0 and count < len(category_files[cat])) or (adjustment < 0 and count > 1)
            ]
            if not adjustable_categories:
                break
            category_to_adjust = max(
                adjustable_categories,
                key=lambda cat: len(category_files[cat]) - files_per_category[cat]
            )
            files_per_category[category_to_adjust] += adjustment
            total_selected += adjustment
        
        # Select files based on calculated counts
        for category, target_num in files_per_category.items():
            files = category_files[category]
            if target_num > 0:
                selected = random.sample(files, min(target_num, len(files)))
                selected_files.extend(selected)
                log_message(f"Selected {len(selected)} files from category {category}", "info")
        
        # Copy selected files
        doc_count = 0
        total_size = 0
        categories_seen = set()
        for group, filename in selected_files:
            src_path = os.path.join(source_dir, group, filename)
            target_filename = f"{group}_{filename}.txt"
            target_path = os.path.join(target_dir, target_filename)
            shutil.copy(src_path, target_path)
            file_size = os.path.getsize(target_path)
            total_size += file_size
            categories_seen.add(group)
            log_message(f"Copied: {target_filename} ({file_size} bytes)", "info")
            print(f"Copied: {target_filename}")
            doc_count += 1
        
        avg_size = total_size / doc_count if doc_count > 0 else 0
        log_message(f"Total documents copied: {doc_count}, Total size: {total_size} bytes, Avg size: {avg_size:.2f} bytes", "info")
        log_message(f"Categories processed: {sorted(list(categories_seen))}", "info")
        print(f"Total documents copied: {doc_count}")
        
        if len(categories_seen) != len(category_files):
            log_message(f"Warning: Only {len(categories_seen)} categories processed out of {len(category_files)}", "warning")
            print(f"Warning: Only {len(categories_seen)} categories processed out of {len(category_files)}")
    except Exception as e:
        log_message(f"Error preparing dataset: {e}", "error")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare 20 Newsgroups dataset")
    parser.add_argument('--source-dir', default="/Users/johnty/Downloads/20news-18828", help="Path to extracted 20 Newsgroups dataset")
    parser.add_argument('--target-dir', default="dataset", help="Target directory for processed files")
    parser.add_argument('--max-docs', type=int, default=None, help="Maximum number of documents to process (default: all)")
    args = parser.parse_args()
    
    prepare_20newsgroups(args.source_dir, args.target_dir, args.max_docs)