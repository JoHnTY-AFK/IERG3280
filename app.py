import os
import sys
import time
from flask import Flask, request, render_template, jsonify
from whoosh.index import open_dir
from whoosh.qparser import QueryParser

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.search import search_index, get_frequent_terms, suggest_query_correction, get_all_categories
from src.utils import log_message

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def search():
    """Handle search queries and display results with pagination, filtering, and suggestions."""
    results_tfidf = []
    results_bm25 = []
    results_custom = []
    total_tfidf = 0
    total_bm25 = 0
    total_custom = 0
    query = ""
    error = None
    suggestion = None
    search_time = 0.0
    current_page = int(request.args.get("page", 1))
    results_per_page = 10
    total_pages = 1  # Define default value to avoid UnboundLocalError
    corrected_query = None

    # Check if index exists
    index_dir = os.path.join("index")
    if not os.path.exists(index_dir) or not os.listdir(index_dir):
        log_message(f"Index directory {index_dir} is empty or does not exist", "error")
        error = f"Index directory {index_dir} is empty or does not exist. Please run index_builder.py."

    # Get autocomplete terms and categories
    autocomplete_terms = get_frequent_terms(index_dir) if not error else []
    categories = get_all_categories(index_dir) if not error else []
    log_message(f"Categories passed to template: {categories}", "info")  # Log categories gale for debugging

    # Handle query from POST or GET
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        category = request.form.get("category", "").strip()
    elif request.method == "GET":
        query = request.args.get("query", "").strip()
        category = request.args.get("category", "").strip()

    if query and not error:
        try:
            start_time = time.time()
            # Suggest correction for all queries
            suggestion = suggest_query_correction(index_dir, query)
            
            # If there's a suggestion, search for the corrected query instead
            search_query = suggestion if suggestion else query
            corrected_query = suggestion if suggestion else None
            
            # Build final query with category filter if provided
            final_query = search_query
            if category:
                final_query = f"{search_query} category:{category}"
            
            # Search with TF-IDF, BM25, and Custom
            all_results_tfidf = search_index(index_dir, final_query, top_n=100, weighting="tfidf")
            all_results_bm25 = search_index(index_dir, final_query, top_n=100, weighting="bm25")
            all_results_custom = search_index(index_dir, final_query, top_n=100, weighting="custom")
            search_time = time.time() - start_time
            log_message(f"Search query '{final_query}' took {search_time:.4f} seconds", "info")

            # Store total counts before pagination
            total_tfidf = len(all_results_tfidf)
            total_bm25 = len(all_results_bm25)
            total_custom = len(all_results_custom)
            log_message(f"Total results - TF-IDF: {total_tfidf}, BM25: {total_bm25}, Custom: {total_custom}", "info")

            # Paginate results
            if all_results_tfidf or all_results_bm25 or all_results_custom:
                max_results = max(total_tfidf, total_bm25, total_custom)
                total_pages = (max_results + results_per_page - 1) // results_per_page
                start_idx = (current_page - 1) * results_per_page
                end_idx = start_idx + results_per_page
                results_tfidf = all_results_tfidf[start_idx:min(end_idx, len(all_results_tfidf))]
                results_bm25 = all_results_bm25[start_idx:min(end_idx, len(all_results_bm25))]
                results_custom = all_results_custom[start_idx:min(end_idx, len(all_results_custom))]

        except Exception as e:
            log_message(f"Error processing query '{query}': {str(e)}", "error")
            error = f"Error processing query: {str(e)}"

    return render_template("search.html", query=query, results_tfidf=results_tfidf,
                         results_bm25=results_bm25, results_custom=results_custom,
                         total_tfidf=total_tfidf, total_bm25=total_bm25, total_custom=total_custom,
                         error=error, suggestion=suggestion, autocomplete_terms=autocomplete_terms,
                         categories=categories, current_page=current_page, total_pages=total_pages,
                         search_time=search_time, corrected_query=corrected_query)

@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    """Return autocomplete suggestions based on the input prefix."""
    prefix = request.args.get("prefix", "").strip()
    index_dir = os.path.join("index")
    try:
        terms = get_frequent_terms(index_dir, top_n=10, prefix=prefix)
        return jsonify(terms)
    except Exception as e:
        log_message(f"Error in autocomplete for prefix '{prefix}': {str(e)}", "error")
        return jsonify([]), 500

@app.route("/document/<doc_id>")
def view_document(doc_id):
    """Display the full content of a document by its ID."""
    try:
        ix = open_dir("index")
        with ix.searcher() as searcher:
            query = QueryParser("doc_id", ix.schema).parse(doc_id)
            results = searcher.search(query, limit=1)
            if results:
                doc = searcher.stored_fields(results[0].docnum)
                return render_template("document.html", doc_id=doc_id, content=doc.get("original_content", ""))
            log_message(f"Document {doc_id} not found in index", "warning")
            return "Document not found", 404
    except Exception as e:
        log_message(f"Error viewing document {doc_id}: {str(e)}", "error")
        return f"Error retrieving document: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)