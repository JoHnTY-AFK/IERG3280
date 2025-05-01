import os
import sys
import random
import time
import pandas as pd
from whoosh.index import open_dir
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search import search_index
from src.utils import log_message

def calculate_ndcg(results, ground_truth, k=10):
    """Calculate Normalized Discounted Cumulative Gain (NDCG) for top k results."""
    relevant = set(ground_truth)
    dcg = 0.0
    idcg = 0.0
    for i, result in enumerate(results[:k], 1):
        if result["doc_id"] in relevant:
            dcg += 1 / (i + 1)  # Simplified relevance score of 1
        if i <= len(relevant):
            idcg += 1 / (i + 1)
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_results(results, ground_truth, query):
    """Calculate precision, recall, F1-score, MAP, and NDCG."""
    relevant = set(ground_truth.get(query, []))
    retrieved = set(result["doc_id"] for result in results)
    true_positives = len(relevant & retrieved)
    precision = true_positives / len(retrieved) if retrieved else 0
    recall = true_positives / len(relevant) if relevant else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    
    # Compute MAP
    ap = 0
    if relevant:
        true_positives = 0
        for i, result in enumerate(results, 1):
            if result["doc_id"] in relevant:
                true_positives += 1
                ap += true_positives / i
        ap = ap / len(relevant) if relevant else 0
    
    # Compute NDCG
    ndcg = calculate_ndcg(results, ground_truth.get(query, []))
    
    return precision, recall, f1, ap, ndcg

def get_sample_terms(index_dir, num_docs=3):
    """Retrieve sample terms from random documents in the index."""
    try:
        ix = open_dir(index_dir)
        with ix.searcher() as searcher:
            reader = searcher.reader()
            all_doc_ids = list(reader.all_doc_ids())
            doc_ids = random.sample(all_doc_ids, min(num_docs, len(all_doc_ids)))
            sample_terms = []
            for doc_num in doc_ids:
                doc_id = searcher.stored_fields(doc_num)["doc_id"]
                content = searcher.stored_fields(doc_num)["content"]
                terms = content.split()[:10]
                sample_terms.append((doc_id, terms))
            return sample_terms
    except Exception as e:
        log_message(f"Error retrieving sample terms: {str(e)}", "error")
        print(f"Error retrieving sample terms: {str(e)}")
        return []

def stress_test(index_dir, queries, num_iterations=100):
    """Simulate high query load to test scalability."""
    start_time = time.time()
    total_queries = 0
    total_results = 0
    
    for _ in range(num_iterations):
        for query in random.sample(queries, len(queries)):
            for weighting in ["tfidf", "bm25", "custom"]:
                results = search_index(index_dir, query, top_n=5, weighting=weighting)
                total_queries += 1
                total_results += len(results)
    
    total_time = time.time() - start_time
    avg_time_per_query = total_time / total_queries if total_queries > 0 else 0
    log_message(f"Stress test: {total_queries} queries, {total_results} results, {total_time:.4f} seconds, Avg time per query: {avg_time_per_query:.6f} seconds", "info")
    print(f"Stress test completed: {total_queries} queries, {total_time:.4f} seconds")

def test_search():
    """Test search functionality and evaluate metrics."""
    index_dir = os.path.join("index")
    
    if not os.path.exists(index_dir) or not os.listdir(index_dir):
        log_message(f"Index directory {index_dir} is empty or does not exist", "error")
        print(f"Error: Index directory {index_dir} is empty or does not exist")
        print("Run: python3 -m src.index_builder")
        return
    
    sample_terms = get_sample_terms(index_dir)
    print("Sample index terms:")
    for doc_id, terms in sample_terms:
        print(f"Document: {doc_id}, Terms: {terms}")
    print()
    
    # Expanded queries
    queries = [
        "comput graphic",
        "hockey",
        "encrypt algorithm",
        "space shuttl",
        "medic diagnosi",
        "comput",
        "graphic",
        "automot",
        "computer graphics",
        "hockey game",
        "render",
        "game",
        '"computer graphics"',
        "comput AND graphic",
        "hockey OR game",
        "neural network",
        "space exploration",
        "cryptography key",
        "baseball",
        '"neural network"'
    ]
    
    # Load ground truth from test_queries.csv
    csv_path = os.path.join("tests", "test_queries.csv")
    ground_truth = {}
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                query = row["query"]
                relevant_docs = row["relevant_docs"]
                if isinstance(relevant_docs, str) and relevant_docs.strip():
                    ground_truth[query] = relevant_docs.split(",")
                else:
                    log_message(f"Skipping invalid relevant_docs for query '{query}': {relevant_docs}", "warning")
            log_message(f"Loaded ground truth from {csv_path}", "info")
            print(f"Loaded ground truth from {csv_path}")
        except Exception as e:
            log_message(f"Error loading test_queries.csv: {str(e)}", "error")
            print(f"Error loading test_queries.csv: {str(e)}")
    
    # Run tests
    for query in queries:
        for weighting in ["tfidf", "bm25", "custom"]:
            log_message(f"Testing query: {query} with {weighting}", "info")
            results = search_index(index_dir, query, top_n=5, weighting=weighting)
            print(f"\nQuery: {query} ({weighting.upper()})")
            if results:
                for result in results:
                    print(f"Document: {result['doc_id']}, Score: {result['score']:.4f}")
                    print(f"Preview: {result['highlight']}")
                precision, recall, f1, map_score, ndcg = evaluate_results(results, ground_truth, query)
                print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, MAP: {map_score:.4f}, NDCG: {ndcg:.4f}")
            else:
                print("No results found.")
            print()
    
    # Run stress test
    stress_test(index_dir, queries)

if __name__ == "__main__":
    test_search()