# 20 Newsgroups Search Engine with Advanced Ranking Methods

## Overview

This project is a search engine for the 20 Newsgroups dataset, developed as part of the IERG3280 course (Networks: Technology, Economics, and Social Interactions) at the Chinese University of Hong Kong. The search engine supports three ranking methods—TF-IDF, BM25, and a Custom Proximity-Based Approach—allowing users to search through 18,846 Usenet posts across 20 categories (e.g., comp.graphics, rec.sport.hockey). It features a user-friendly web interface with pagination, category filtering, autocomplete, and query suggestion capabilities, achieving sub-second search times (< 0.2s).

The project explores how search engines facilitate information access in networked environments, aligning with IERG3280's focus on network technologies and social interactions. It serves as a valuable tool for information retrieval (IR) education, demonstrating practical implementations of ranking algorithms and user-facing features.

## Features

- **Three Ranking Methods**:
  - **TF-IDF**: Weights terms based on frequency and rarity across documents.
  - **BM25**: Improves ranking with term saturation and document length normalization.
  - **Custom Proximity**: Boosts scores for documents where query terms appear closer together, ideal for contextual queries (e.g., "computer graphics").
- **Full Dataset Indexing**: Indexes all 20 categories of the 20 Newsgroups dataset using proportional sampling to ensure balanced representation.
- **Fast Search**: Achieves average search times of 0.12s (TF-IDF), 0.14s (BM25), and 0.18s (Custom), all under the 0.2s target.
- **Web Interface**:
  - Built with Flask, Tailwind CSS, and JavaScript.
  - Features include pagination (10 results per page), category filtering, autocomplete for query terms, and query suggestion for misspellings (e.g., "mural network" → "neural network").
  - Document view to display full content of search results.
- **Stem Mapping**: Maps stemmed terms (e.g., "comput") to unstemmed forms (e.g., "computer") for improved autocomplete and query suggestion.
- **Testing**: Includes unit tests to validate search functionality.

## Project Structure

The project is organized as follows:

```
IERG3280_PROJECT/
├── dataset/                     # Raw 20 Newsgroups dataset files
├── dataset_processed/           # Preprocessed dataset files (stemmed, stopwords removed)
├── docs/                        # Documentation files
│   ├── case_studies.md          # Case studies on search engine usage
│   ├── concepts_summary.md      # Summary of IR concepts used
│   ├── design_document.md       # System design details
│   ├── project_report.md        # Full project report
│   └── social_analysis.md       # Analysis of social interactions in the dataset
├── index/                       # Whoosh index directory
├── src/                         # Source code for core functionality
│   ├── __init__.py              # Package initialization
│   ├── index_builder.py         # Indexes dataset and builds stem mapping
│   ├── prepare_20newsgroups.py  # Fetches and prepares the 20 Newsgroups dataset
│   ├── preprocess.py            # Preprocesses text (lowercase, stemming, stopwords)
│   ├── search.py                # Search logic with ranking methods
│   └── utils.py                 # Utility functions (e.g., logging)
├── static/                      # Static assets for web interface
│   └── search.css               # Custom CSS for styling
├── templates/                   # HTML templates for Flask
│   ├── document.html            # Template for viewing full documents
│   └── search.html              # Main search interface template
├── tests/                       # Test files
│   ├── test_queries.csv         # Sample queries for testing
│   └── test_search.py           # Unit tests for search functionality
├── venv/                        # Virtual environment directory
├── .gitignore                   # Git ignore file
├── app.py                       # Flask application entry point
├── commands.txt                 # List of setup and run commands
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── search_engine.log            # Log file for debugging
├── setup.py                     # Script to set up virtual environment
└── structure.txt                # Project structure documentation
```

## Prerequisites

To run this project, ensure you have the following installed:

- **Python 3.9+**: The project is built with Python 3.9.
- **pip**: For installing dependencies.
- **Virtual Environment**: Recommended to isolate dependencies.
- **Internet Connection**: Required to download the 20 Newsgroups dataset and NLTK data during setup.

## Installation

Follow these steps to set up the project on your local machine:

1. **Clone the Repository** (if using version control):
   ```bash
   git clone <repository-url>
   cd IERG3280_PROJECT
   ```

2. **Install Dependencies**:
   Install the required Python packages listed in `requirements.txt`:
   ```bash
   pip3 install -r requirements.txt
   ```
   The dependencies include:
   - `whoosh==2.7.4`: For indexing and searching.
   - `flask==2.0.1`: Web framework for the interface.
   - `pandas>=1.5.0`: For data handling (if needed in future extensions).
   - `nltk==3.6.3`: For text preprocessing (stemming, stopwords).
   - `jinja2==3.0.1`: Templating engine for Flask.
   - `werkzeug<3.0.0`: WSGI utility for Flask.
   - `pyspellchecker>=0.7.0`: For query suggestion.

3. **Set Up the Virtual Environment**:
   Create and activate a virtual environment to isolate the project’s dependencies:
   ```bash
   python3 setup.py
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

4. **Download NLTK Data**:
   The project uses NLTK for preprocessing. Download the required NLTK data (e.g., stopwords, tokenizers):
   ```bash
   python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

## Usage

Follow these steps to prepare the dataset, build the index, test the search functionality, and launch the web interface.

### Step 1: Prepare the Dataset
Fetch the 20 Newsgroups dataset and prepare it for processing:
```bash
python3 -m src.prepare_20newsgroups
```
- This script uses `fetch_20newsgroups` from scikit-learn to download the dataset.
- It applies proportional sampling to ensure all 20 categories are represented, even if a document limit is set.
- Output: Raw documents are saved as `.txt` files in the `dataset` directory (e.g., `comp.graphics_58488.txt`).

### Step 2: Preprocess the Dataset
Preprocess the raw documents to make them suitable for indexing:
```bash
python3 -m src.preprocess
```
- This script:
  - Converts text to lowercase.
  - Removes punctuation using regex.
  - Tokenizes text using NLTK’s `word_tokenize`.
  - Removes stopwords using NLTK’s English stopwords list.
  - Applies stemming with NLTK’s `PorterStemmer` (e.g., "computing" → "comput").
- Output: Preprocessed documents are saved in the `dataset_processed` directory.

### Step 3: Build the Index
Index the preprocessed documents and create a stem mapping for autocomplete and query suggestion:
```bash
python3 -m src.index_builder
```
- This script:
  - Uses Whoosh to create an index in the `index` directory.
  - Defines a schema with fields: `doc_id`, `content` (preprocessed), `original_content` (raw), and `category`.
  - Builds a stem mapping (`stem_mapping.pkl`) to map stemmed terms to unstemmed forms (e.g., "comput" → {"computer", "computing"}).
  - Handles encoding errors by falling back to `latin-1` if UTF-8 decoding fails.
- Output: Indexed documents in the `index` directory and `stem_mapping.pkl` for autocomplete/query suggestion.

### Step 4: Test the Search Functionality
Run unit tests to validate the search engine:
```bash
python3 -m tests.test_search
```
- This script:
  - Loads test queries from `test_queries.csv` (e.g., "computer graphics", "hockey game").
  - Tests search functionality for TF-IDF, BM25, and Custom Proximity methods.
  - Verifies that results are returned and scores are reasonable.
- Output: Test results in the terminal (e.g., "All tests passed").

### Step 5: Run a Command-Line Search (Optional)
Perform a search directly from the command line to debug or test:
```bash
python3 -m src.search
```
- This script:
  - Prompts for a query (e.g., "computer graphics").
  - Searches the index using the default TF-IDF ranking.
  - Returns a list of document IDs and scores.
- Output: Search results in the terminal.

### Step 6: Launch the Web Interface
Start the Flask web server to access the search engine via a browser:
```bash
python3 app.py
```
- This script:
  - Launches a Flask server on `localhost:5000`.
  - Provides a web interface to search the dataset, filter by category, and view documents.
- Open your browser and navigate to `http://localhost:5000`.

## Web Interface Usage

Once the Flask server is running, you can interact with the search engine through the web interface at `http://localhost:5000`.

### Features
- **Search**: Enter a query (e.g., "computer graphics") to search the dataset. Results are displayed in three tabs:
  - TF-IDF: Traditional frequency-based ranking.
  - BM25: Enhanced ranking with term saturation.
  - Custom Proximity: Prioritizes documents where query terms are closer together.
- **Category Filtering**: Use the dropdown to filter results by category (e.g., "comp.graphics").
- **Pagination**: Navigate through results with 10 documents per page.
- **Autocomplete**: Start typing a query (e.g., "comp"), and suggestions like "computer" or "computing" will appear.
- **Query Suggestion**: Misspelled queries (e.g., "mural network") are corrected (e.g., "neural network") with a clickable message: "These are results for neural network. Search instead for mural network."
- **Document View**: Click "View Full Document" to see the raw content of a result (e.g., `rec.sport.hockey_53967.txt`).

### Example Usage
1. Search for "computer graphics":
   - Results will show documents containing these terms, ranked by each method.
   - The Custom Proximity tab might prioritize a document where "computer" and "graphics" appear in the same sentence.
2. Filter by "comp.graphics" to narrow down results.
3. Try "mural network" to see the query suggestion feature correct it to "neural network."

## Implementation Details

### Data Preparation (`src/prepare_20newsgroups.py`)
- Fetches the 20 Newsgroups dataset using `fetch_20newsgroups` from scikit-learn.
- Removes headers, footers, and quotes to clean the data.
- Implements proportional sampling to ensure all 20 categories are represented:
  - Calculates the proportion of documents per category.
  - Samples documents proportionally if a `max_docs` limit is set.
- Saves raw documents as `.txt` files in the `dataset` directory.

### Preprocessing (`src/preprocess.py`)
- Processes raw documents to prepare them for indexing:
  - Converts text to lowercase.
  - Removes punctuation with regex (`[^\w\s]`).
  - Tokenizes with NLTK’s `word_tokenize`.
  - Removes stopwords using NLTK’s English stopwords.
  - Applies stemming with `PorterStemmer` (e.g., "running" → "run").
- Saves preprocessed documents in the `dataset_processed` directory.

### Indexing (`src/index_builder.py`)
- Uses Whoosh to create an index with the following schema:
  - `doc_id`: Unique identifier (stored).
  - `content`: Preprocessed text (indexed and stored).
  - `original_content`: Raw text (stored for display).
  - `category`: Document category (stored for filtering).
- Builds a stem mapping in a single pass:
  - Maps stemmed terms to unstemmed forms (e.g., "comput" → {"computer", "computing"}).
  - Saved as `stem_mapping.pkl` in the `index` directory.
- Handles encoding errors by falling back to `latin-1` if UTF-8 fails.

### Search (`src/search.py`)
- Implements three ranking methods:
  - **TF-IDF**: Uses Whoosh’s `TF_IDF` scoring.
  - **BM25**: Uses Whoosh’s `BM25F` scoring with default parameters.
  - **Custom Proximity**: Adjusts scores based on the proximity of query terms in the document (closer terms = higher score).
    - Calculates the minimum distance between query terms in the document.
    - Boosts the score if the distance is within a threshold (e.g., 10 words).
- Supports query preprocessing:
  - Matches the preprocessing steps (lowercase, stemming, stopwords).
  - Preserves phrase queries (e.g., `"neural network"`) and boolean queries (e.g., "hockey AND game").
- Includes query suggestion:
  - Uses `pyspellchecker` with a custom word list (e.g., "neural", "network").
  - Falls back to `SequenceMatcher` for similarity matching (threshold: 0.3).
- Highlights query terms in results using Whoosh’s highlighter.

### Web Interface (`app.py`, `templates/`, `static/`)
- Built with Flask, Tailwind CSS, and JavaScript.
- **Routes**:
  - `/`: Main search page with results, pagination, and category filtering.
  - `/autocomplete`: Returns autocomplete suggestions based on the query prefix.
  - `/document/<doc_id>`: Displays the full content of a document.
- **Templates**:
  - `search.html`: Main interface with search bar, tabs for ranking methods, and results.
  - `document.html`: Displays the raw content of a document.
- **Styling**:
  - Uses Tailwind CSS for a responsive, modern design.
  - Custom styles in `search.css` for additional tweaks (e.g., loading spinner).

### Testing (`tests/test_search.py`)
- Unit tests to validate search functionality.
- Loads sample queries from `test_queries.csv`.
- Tests each ranking method to ensure results are returned and scores are non-zero.
- Example test: Search for "computer graphics" and verify that at least one result is returned.

## Performance Metrics

- **Search Times (Average)**:
  - TF-IDF: 0.12 seconds
  - BM25: 0.14 seconds
  - Custom Proximity: 0.18 seconds
- **Indexing Time**: ~25 seconds for 18,846 documents.
- **Memory Usage**: ~150 MB for the index and Flask server.
- **User Study** (10 participants):
  - Average time to find relevant documents: 15 seconds.
  - Feedback: Users appreciated the category filter, tabbed interface, and query suggestion feature.
  - Ranking Effectiveness: Custom Proximity preferred for contextual queries (70% user preference for queries like "computer graphics").

## Challenges and Solutions

### Challenge 1: Incomplete Category Coverage
- **Issue**: Initially, only 11/20 categories were indexed due to a document limit, reducing search relevance.
- **Solution**: Implemented proportional sampling in `prepare_20newsgroups.py` to ensure all categories are represented.
- **Result**: Increased coverage from 55% to 100%, as validated by the category dropdown in the web interface.

### Challenge 2: Encoding Errors in Dataset
- **Issue**: Some files in the 20 Newsgroups dataset contained non-UTF-8 characters, causing decoding errors during indexing.
- **Solution**: Added error handling in `index_builder.py` to fall back to `latin-1` encoding if UTF-8 fails.
- **Result**: Indexing completes successfully for all files.

### Challenge 3: Slow Stem Mapping
- **Issue**: Building the stem mapping took over 10 minutes due to a nested loop comparing all stemmed and unstemmed terms.
- **Solution**: Optimized `index_builder.py` to build the stem mapping in a single pass while indexing, using a stemming cache to avoid redundant operations.
- **Result**: Reduced indexing time to ~25 seconds.

### Challenge 4: Query Suggestion for Misspellings
- **Issue**: Misspelled queries like "mural network" weren’t corrected to "neural network" due to limitations in `pyspellchecker`.
- **Solution**: Enhanced `search.py` with a custom word list ("neural", "network") and lowered the similarity threshold to 0.3.
- **Result**: Successfully suggests "neural network" with a clickable UI message.

## Limitations

- **Advanced Query Syntax**: The search engine does not yet support wildcards (e.g., "comp*") or full phrase query handling beyond exact matches.
- **Scalability**: The system is not optimized for distributed search, which would be necessary for larger datasets.
- **Ranking for Single-Term Queries**: Custom Proximity is less effective for single-term queries compared to TF-IDF or BM25.

## Future Work

- **Advanced Query Syntax**: Add support for wildcards, phrase queries, and more complex boolean expressions.
- **Distributed Search**: Explore frameworks like Elasticsearch for scalability with larger datasets.
- **Improved Ranking**: Incorporate machine learning-based ranking methods (e.g., learning-to-rank) for better relevance.
- **User Feedback Integration**: Add a feedback mechanism in the UI to allow users to rate results, improving future rankings.

## Documentation

Additional documentation is available in the `docs` directory:
- `case_studies.md`: Examples of search engine usage in real-world scenarios.
- `concepts_summary.md`: Overview of IR concepts like TF-IDF, BM25, and proximity scoring.
- `design_document.md`: Detailed system design and architecture.
- `project_report.md`: Full project report with results, user study, and figures.
- `social_analysis.md`: Analysis of social interactions in the 20 Newsgroups dataset.

## Troubleshooting

- **UnicodeDecodeError During Indexing**:
  - Ensure `index_builder.py` has the encoding fallback (`latin-1`) enabled.
  - Check `search_engine.log` for the problematic file and re-encode it to UTF-8 using `iconv`:
    ```bash
    iconv -f WINDOWS-1252 -t UTF-8 dataset/<filename> -o dataset/<filename>.utf8
    ```
- **No Results for a Query**:
  - Verify the `index` directory exists and contains indexed documents.
  - Run `python3 -m src.index_builder` to rebuild the index.
- **Flask Server Fails to Start**:
  - Ensure you’re in the virtual environment (`source venv/bin/activate`).
  - Check if port 5000 is in use and change it in `app.py` if needed.
- **Autocomplete/Query Suggestion Not Working**:
  - Ensure `stem_mapping.pkl` exists in the `index` directory.
  - Rebuild the index if missing.

## Contributing

This project was developed for IERG3280 and is not actively maintained. However, contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

## Acknowledgments

- **IERG3280 Course**: For providing the framework and motivation for this project.
- **20 Newsgroups Dataset**: A widely-used benchmark dataset for IR research.
- **xAI**: For inspiration and guidance on building AI-driven tools (e.g., through interactions with Grok).
- **Libraries**: Thanks to the developers of Whoosh, Flask, NLTK, and pyspellchecker for their excellent tools.

## Contact

For questions or feedback, please contact:  
Tsoi Ming Hon  
[Your Email or Preferred Contact Method]

---

*Last updated: May 1, 2025*