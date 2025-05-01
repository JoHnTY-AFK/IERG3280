import os
import sys
import time
import traceback
import pickle
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from whoosh.index import open_dir
from whoosh.qparser import MultifieldParser, OrGroup, QueryParser
from whoosh import scoring, highlight
from whoosh.searching import NoTermsException
from src.utils import log_message
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import re
from difflib import SequenceMatcher
from spellchecker import SpellChecker

def preprocess_query(query_str, remove_stopwords=True, for_search=True):
    """Preprocess query to match document preprocessing, returning both stemmed and raw tokens."""
    try:
        if for_search and query_str.startswith('"') and query_str.endswith('"'):
            return query_str[1:-1], [query_str[1:-1].lower()]
        
        stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        ps = PorterStemmer()
        
        query_str = re.sub(r'[^\w\s]', '', query_str)
        tokens = word_tokenize(query_str.lower())
        raw_tokens = tokens.copy()
        tokens = [t for t in tokens if t not in string.punctuation]
        if remove_stopwords and not (' AND ' in query_str or ' OR ' in query_str):
            tokens = [t for t in tokens if t not in stop_words]
        stemmed_tokens = [ps.stem(t) for t in tokens]
        
        processed_query = " ".join(stemmed_tokens)
        return processed_query if processed_query.strip() else query_str, raw_tokens
    except Exception as e:
        log_message(f"Query preprocessing error for '{query_str}': {str(e)}", "error")
        log_message(traceback.format_exc(), "error")
        print(f"Error preprocessing query '{query_str}': {str(e)}")
        return query_str, [query_str.lower()]

def generate_term_variations(term, stemmer, cache=None):
    """Generate possible variations of a term, using a cache if provided."""
    if cache is None:
        cache = {}
    term_lower = term.lower()
    if term_lower in cache:
        return cache[term_lower]
    
    variations = [term_lower]
    stemmed = stemmer.stem(term_lower)
    if stemmed != term_lower:
        variations.append(stemmed)
    suffixes = ['s', 'ed', 'ing', 'ion', 'ive', 'er']
    for suffix in suffixes:
        if not term_lower.endswith(suffix):
            variations.append(term_lower + suffix)
    variations = list(set(variations))
    cache[term_lower] = variations
    return variations

def highlight_keywords(text, query_terms, stemmer, variation_cache=None):
    """Highlight only the exact query terms in the text, using cached variations."""
    if variation_cache is None:
        variation_cache = {}
    highlighted_text = text
    for term in query_terms:
        term_lower = term.lower()
        variations = generate_term_variations(term, stemmer, variation_cache)
        variations = sorted(variations, key=len, reverse=True)
        for variation in variations:
            variation_lower = variation.lower()
            pos = variation_lower.find(term_lower)
            if pos == -1:
                continue
            before = re.escape(variation_lower[:pos])
            after = re.escape(variation_lower[pos + len(term_lower):])
            pattern = re.compile(
                rf'({before})({re.escape(term_lower)})({after})',
                re.IGNORECASE
            )
            highlighted_text = pattern.sub(
                rf'\1<b class="bg-yellow-200">\2</b>\3',
                highlighted_text
            )
    return highlighted_text

def calculate_proximity_boost(content, query_terms, max_distance=10):
    """Calculate a score boost based on term proximity and query position."""
    if len(query_terms) < 2:
        return 1.0
    
    words = content.split()
    min_distance = float('inf')
    term_positions = {term: [] for term in query_terms}
    
    for i, word in enumerate(words):
        for term in query_terms:
            if word.lower() == term.lower():
                term_positions[term].append(i)
    
    # Weight terms by query position (earlier terms get higher weight)
    term_weights = {term: 1.0 / (i + 1) for i, term in enumerate(query_terms)}
    
    for i, term1 in enumerate(query_terms):
        for term2 in query_terms[i+1:]:
            pos1 = term_positions.get(term1, [])
            pos2 = term_positions.get(term2, [])
            for p1 in pos1:
                for p2 in pos2:
                    distance = abs(p1 - p2)
                    if distance < min_distance:
                        min_distance = distance
    
    if min_distance <= max_distance:
        # Combine proximity and term weights
        weight_sum = sum(term_weights[term] for term in query_terms)
        boost = 1.0 + (max_distance - min_distance) / max_distance * (weight_sum / len(query_terms))
    else:
        boost = 1.0
    
    log_message(f"Proximity boost for terms {query_terms}: {boost:.4f} (min_distance={min_distance})", "info")
    return boost

def clean_snippet(text, query_terms, stemmer, max_length=150, variation_cache=None):
    """Generate a snippet prioritizing multiple query terms."""
    if variation_cache is None:
        variation_cache = {}
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    text_lower = text.lower()
    
    # Split text into potential snippets (windows of ~100 characters)
    words = text.split()
    window_size = 20
    snippets = []
    for i in range(0, len(words), window_size // 2):
        window = words[i:i + window_size]
        snippet_text = ' '.join(window)
        if len(snippet_text) > max_length:
            snippet_text = snippet_text[:max_length]
        # Score snippet based on query term occurrences
        score = sum(1 for term in query_terms if term.lower() in snippet_text.lower())
        snippets.append((snippet_text, score, i))
    
    # Sort snippets by score (more query terms) and position (earlier is better)
    snippets.sort(key=lambda x: (-x[1], x[2]))
    
    if snippets and snippets[0][1] > 0:
        snippet = snippets[0][0]
        if snippets[0][2] > 0:
            snippet = "..." + snippet
        if len(snippet) < len(text):
            snippet = snippet + "..."
        return highlight_keywords(snippet, query_terms, stemmer, variation_cache)
    
    # Fallback: First max_length characters
    snippet = text[:max_length] + "..." if len(text) > max_length else text
    return highlight_keywords(snippet, query_terms, stemmer, variation_cache).strip() or "No preview available"

def get_frequent_terms(index_dir, top_n=10, prefix=None):
    """Retrieve frequent terms from the index for autocompletion, using stem mapping."""
    try:
        ix = open_dir(index_dir)
        # Load stem mapping
        mapping_file = os.path.join(index_dir, "stem_mapping.pkl")
        if not os.path.exists(mapping_file):
            log_message(f"Stem mapping file {mapping_file} not found", "error")
            return []
        
        with open(mapping_file, 'rb') as f:
            stem_mapping = pickle.load(f)
        
        with ix.searcher() as searcher:
            reader = searcher.reader()
            terms = []
            # Fetch more terms to ensure better coverage
            for fieldname, term in reader.most_frequent_terms("content", number=1000):
                term_str = term.decode('utf-8')
                # Only include terms that match the prefix
                if prefix and not term_str.lower().startswith(prefix.lower()):
                    continue
                # Map stemmed term back to unstemmed forms
                if term_str in stem_mapping:
                    unstemmed_terms = stem_mapping[term_str]
                    # Filter unstemmed terms by prefix and add them
                    for unstemmed in unstemmed_terms:
                        if prefix and not unstemmed.lower().startswith(prefix.lower()):
                            continue
                        terms.append(unstemmed)
                else:
                    terms.append(term_str)
            
            # Remove duplicates, sort by length and frequency
            terms = list(set(terms))
            # Prioritize longer, more meaningful terms
            terms = sorted(terms, key=lambda x: (len(x), x))
            terms = terms[:top_n]
            log_message(f"Fetched {len(terms)} terms for prefix '{prefix}': {terms}", "info")
            return terms
    except Exception as e:
        log_message(f"Error retrieving frequent terms: {str(e)}", "error")
        return []

def get_all_categories(index_dir):
    """Retrieve all unique categories from the index using stored fields."""
    try:
        log_message(f"Attempting to retrieve categories from index: {index_dir}", "info")
        ix = open_dir(index_dir)
        categories = set()
        
        with ix.searcher() as searcher:
            doc_count = ix.doc_count()
            log_message(f"Index contains {doc_count} documents", "info")
            for docnum in range(doc_count):
                stored_fields = searcher.stored_fields(docnum)
                if 'category' in stored_fields and stored_fields['category']:
                    category = stored_fields['category']
                    categories.add(category)
                    log_message(f"Found category: {category} in document {docnum}", "info")
                else:
                    log_message(f"No category found in document {docnum}", "warning")
        
        if not categories:
            log_message("No categories found in index", "warning")
        else:
            log_message(f"Retrieved {len(categories)} unique categories: {sorted(list(categories))}", "info")
        
        return sorted(list(categories))
    except Exception as e:
        log_message(f"Error retrieving categories: {str(e)}", "error")
        log_message(traceback.format_exc(), "error")
        return []

def suggest_query_correction(index_dir, query, max_suggestions=1):
    """Suggest a corrected query using spell checking and index terms."""
    try:
        # Initialize spell checker with a custom word list
        spell = SpellChecker()
        # Add domain-specific terms to the spell checker's dictionary
        custom_terms = [
            'neural', 'network', 'computer', 'graphics', 'hockey', 'game',
            'space', 'science', 'religion', 'politics', 'medicine', 'technology'
        ]  # Add more terms relevant to 20 Newsgroups
        spell.word_frequency.load_words(custom_terms)
        
        # Load index terms
        ix = open_dir(index_dir)
        with ix.searcher() as searcher:
            reader = searcher.reader()
            all_terms = set(term.decode('utf-8') for _, term in reader.all_terms() if term[0] == 'content')
        
        # Load stem mapping
        mapping_file = os.path.join(index_dir, "stem_mapping.pkl")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'rb') as f:
                stem_mapping = pickle.load(f)
        else:
            stem_mapping = {}
        
        query_tokens = query.lower().split()
        suggestions = []
        
        for token in query_tokens:
            # First, try spell checker
            corrected = spell.correction(token)
            if corrected != token and (corrected in all_terms or corrected in custom_terms):
                suggestions.append(corrected)
                continue
            
            # If spell checker fails, fall back to similarity matching with unstemmed terms
            best_match = None
            best_similarity = 0.3  # Lowered threshold to 0.3
            for term in all_terms:
                similarity = SequenceMatcher(None, token, term).ratio()
                if similarity > best_similarity:
                    best_match = term
                    best_similarity = similarity
            
            # Also check custom terms
            for term in custom_terms:
                similarity = SequenceMatcher(None, token, term).ratio()
                if similarity > best_similarity:
                    best_match = term
                    best_similarity = similarity
            
            if best_match and best_similarity >= 0.3:
                suggestions.append(best_match)
            else:
                # Try unstemmed terms from stem mapping
                ps = PorterStemmer()
                stemmed_token = ps.stem(token)
                if stemmed_token in stem_mapping:
                    unstemmed_terms = stem_mapping[stemmed_token]
                    for unstemmed in unstemmed_terms:
                        similarity = SequenceMatcher(None, token, unstemmed).ratio()
                        if similarity > best_similarity:
                            best_match = unstemmed
                            best_similarity = similarity
                if best_match and best_similarity >= 0.3:
                    suggestions.append(best_match)
                else:
                    suggestions.append(token)
        
        if suggestions:
            corrected_query = ' '.join(suggestions[:len(query_tokens)])
            if corrected_query.lower() != query.lower():
                log_message(f"Suggested correction for '{query}': '{corrected_query}' (similarity={best_similarity:.2f})", "info")
                return corrected_query
        return None
    except Exception as e:
        log_message(f"Error suggesting query correction: {str(e)}", "error")
        return None

def search_index(index_dir, query_str, top_n=50, weighting="tfidf"):
    """Search the index for a query and return top results with highlights."""
    start_time = time.time()
    try:
        if not query_str.strip():
            log_message("Empty query provided", "warning")
            print("Warning: Empty query provided")
            return []

        if not os.path.exists(index_dir) or not os.listdir(index_dir):
            log_message(f"Index directory {index_dir} is empty or does not exist", "error")
            print(f"Error: Index directory {index_dir} is empty or does not exist")
            return []

        ix = open_dir(index_dir)
        
        doc_count = ix.doc_count()
        log_message(f"Index contains {doc_count} documents", "info")
        print(f"Index contains {doc_count} documents")
        if doc_count == 0:
            log_message("Index is empty", "error")
            print("Error: Index is empty")
            return []
        
        is_phrase_query = query_str.startswith('"') and query_str.endswith('"')
        is_boolean_query = ' AND ' in query_str or ' OR ' in query_str
        
        stemmer = PorterStemmer()
        variation_cache = {}  # Cache for term variations
        processed_query, raw_tokens = preprocess_query(query_str, remove_stopwords=not (is_phrase_query or is_boolean_query), for_search=True)
        highlight_query = preprocess_query(query_str, remove_stopwords=False, for_search=False)[0]
        query_terms = raw_tokens
        log_message(f"Original query: {query_str}, Processed query (search): {processed_query}, Highlight query: {highlight_query}, Query terms: {query_terms}", "info")
        print(f"Query: {query_str} -> Processed (search): {processed_query} -> Highlight: {highlight_query}")
        
        if weighting == "bm25":
            scoring_model = scoring.BM25F()
        elif weighting == "custom":
            scoring_model = scoring.TF_IDF()
        else:
            scoring_model = scoring.TF_IDF()
        
        with ix.searcher(weighting=scoring_model) as searcher:
            category_boost = ""
            if "space" in query_str.lower():
                category_boost = "category:sci.space^2"
            
            if category_boost:
                if is_phrase_query or is_boolean_query:
                    final_query_str = f"({query_str}) {category_boost}"
                else:
                    final_query_str = f"{processed_query} {category_boost}"
            else:
                final_query_str = query_str if (is_phrase_query or is_boolean_query) else processed_query
            
            if is_phrase_query:
                query = QueryParser("content", ix.schema).parse(processed_query)
                if category_boost:
                    query = QueryParser("content", ix.schema).parse(f"({processed_query}) {category_boost}")
                log_message(f"Phrase query parsed: {query}", "info")
                print(f"Phrase query parsed: {query}")
            elif is_boolean_query:
                query = QueryParser("content", ix.schema).parse(final_query_str)
                log_message(f"Boolean query parsed: {query}", "info")
                print(f"Boolean query parsed: {query}")
            else:
                query = MultifieldParser(["content", "doc_id", "category"], ix.schema, group=OrGroup, fieldboosts={"category": 2.0}).parse(final_query_str)
                log_message(f"Standard query parsed: {query}", "info")
                print(f"Parsed query: {query}")
            
            formatter = highlight.HtmlFormatter(tagname="b", classname="bg-yellow-200")
            results = searcher.search(query, limit=top_n, terms=True)
            results.formatter = formatter
            
            matched_terms = []
            if results:
                try:
                    matched_terms = [term[1] for term in results.matched_terms()]
                except NoTermsException:
                    matched_terms = []
            log_message(f"Matched terms: {matched_terms}", "info")
            print(f"Matched terms: {matched_terms}")
            
            result_list = []
            if weighting == "custom":
                query_terms_processed = highlight_query.lower().split()
                for r in results:
                    content = searcher.stored_fields(r.docnum)["content"]
                    boost = calculate_proximity_boost(content, query_terms_processed)
                    adjusted_score = r.score * boost
                    result_list.append({
                        "doc_id": r["doc_id"],
                        "score": adjusted_score,
                        "content": content,
                        "original_content": searcher.stored_fields(r.docnum).get("original_content", content)
                    })
                result_list = sorted(result_list, key=lambda x: x["score"], reverse=True)[:top_n]
            else:
                result_list = [
                    {
                        "doc_id": r["doc_id"],
                        "score": r.score,
                        "content": searcher.stored_fields(r.docnum)["content"],
                        "original_content": searcher.stored_fields(r.docnum).get("original_content", searcher.stored_fields(r.docnum)["content"])
                    }
                    for r in results
                ]
            
            # Wildcard search with prefix and suffix
            if not result_list and not (is_phrase_query or is_boolean_query):
                wildcard_query_str = " ".join([f"*{term}*" for term in query_terms])
                if category_boost:
                    wildcard_query_str = f"{wildcard_query_str} {category_boost}"
                query = MultifieldParser(["content", "doc_id", "category"], ix.schema, group=OrGroup, fieldboosts={"category": 2.0}).parse(wildcard_query_str)
                log_message(f"Wildcard query parsed: {query}", "info")
                print(f"Wildcard query parsed: {query}")
                
                results = searcher.search(query, limit=top_n, terms=True)
                results.formatter = formatter
                
                matched_terms = []
                if results:
                    try:
                        matched_terms = [term[1] for term in results.matched_terms()]
                    except NoTermsException:
                        matched_terms = []
                log_message(f"Wildcard matched terms: {matched_terms}", "info")
                print(f"Wildcard matched terms: {matched_terms}")
                
                result_list = [
                    {
                        "doc_id": r["doc_id"],
                        "score": r.score,
                        "content": searcher.stored_fields(r.docnum)["content"],
                        "original_content": searcher.stored_fields(r.docnum).get("original_content", searcher.stored_fields(r.docnum)["content"])
                    }
                    for r in results
                ]
            
            try:
                final_results = []
                for r in result_list:
                    original_content = r["original_content"]
                    if not original_content.strip():
                        log_message(f"Empty original_content for {r['doc_id']}", "warning")
                        highlight_text = "No preview available"
                    else:
                        try:
                            highlight_text = results.highlights("original_content", text=original_content, top=1, minscore=0, maxchars=300, surround=50)
                            if not highlight_text:
                                highlight_text = clean_snippet(original_content, query_terms, stemmer, variation_cache=variation_cache)
                                log_message(f"Highlight empty for {r['doc_id']}: Using cleaned snippet", "warning")
                            highlight_text = highlight_keywords(highlight_text, query_terms, stemmer, regex=True)
                        except Exception as e:
                            log_message(f"Highlight error for {r['doc_id']}: {str(e)}", "warning")
                            highlight_text = clean_snippet(original_content, query_terms, stemmer, variation_cache=variation_cache)
                    final_results.append({
                        "doc_id": r["doc_id"],
                        "score": r["score"],
                        "highlight": highlight_text
                    })
                result_list = final_results
            except Exception as e:
                log_message(f"Highlighting error for query '{query_str}': {str(e)}", "error")
                log_message(traceback.format_exc(), "error")
                result_list = [
                    {
                        "doc_id": r["doc_id"],
                        "score": r["score"],
                        "highlight": clean_snippet(r["original_content"], query_terms, stemmer, variation_cache=variation_cache)
                    }
                    for r in result_list
                ]
                log_message(f"Highlighting failed for query '{query_str}': Using cleaned snippets", "warning")
            
            if result_list:
                sample_docs = [r["doc_id"] for r in result_list[:3]]
                log_message(f"Sample matched documents: {sample_docs}", "info")
                print(f"Sample matched documents: {sample_docs}")
            
            log_message(f"Query: {query_str}, Weighting: {weighting}, Results: {len(result_list)}", "info")
            if not result_list:
                raw_query = MultifieldParser(["content", "doc_id", "category"], ix.schema, group=OrGroup).parse(query_str)
                log_message(f"Fallback raw query: {raw_query}", "info")
                print(f"Trying raw query: {raw_query}")
                results = searcher.search(raw_query, limit=top_n, terms=True)
                results.formatter = formatter
                matched_terms = []
                if results:
                    try:
                        matched_terms = [term[1] for term in results.matched_terms()]
                    except NoTermsException:
                        matched_terms = []
                log_message(f"Raw query matched terms: {matched_terms}", "info")
                print(f"Raw query matched terms: {matched_terms}")
                result_list = [
                    {
                        "doc_id": r["doc_id"],
                        "score": r.score,
                        "highlight": r.highlights("original_content", top=1, minscore=0, maxchars=300, surround=50) or clean_snippet(searcher.stored_fields(r.docnum).get("original_content", searcher.stored_fields(r.docnum)["content"]), query_terms, stemmer, variation_cache=variation_cache)
                    }
                    for r in results
                ]
                for r in result_list:
                    r["highlight"] = highlight_keywords(r["highlight"], query_terms, stemmer, variation_cache)
                if result_list:
                    log_message(f"Raw query results: {len(result_list)}", "info")
                    print(f"Raw query found {len(result_list)} results")
            
            if not result_list:
                print(f"No results found for query '{query_str}' ({weighting.upper()})")
            
            search_time = time.time() - start_time
            log_message(f"Search completed in {search_time:.4f} seconds for query '{query_str}' ({weighting})", "info")
            return result_list
    except NoTermsException:
        log_message(f"No terms found in index for query '{query_str}'", "error")
        print(f"No terms found in index for query '{query_str}'")
        return []
    except Exception as e:
        log_message(f"Search error for query '{query_str}': {str(e)}", "error")
        log_message(traceback.format_exc(), "error")
        print(f"Search error for query '{query_str}': {str(e)}")
        return []