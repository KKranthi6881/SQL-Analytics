import streamlit as st
from pathlib import Path
import os
from utils import ChromaDBManager
from processor import SearchProcessor
import yaml
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize managers
BASE_DIR = Path(__file__).resolve().parent.parent
db_manager = ChromaDBManager(persist_directory=str(BASE_DIR / "chroma_db"))
search_processor = SearchProcessor(db_manager)

# Set page configuration
st.set_page_config(
    page_title="Document & Code Search Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e6f3ff;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
    }
    .search-result {
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .similarity-score {
        color: #388e3c;
        font-weight: bold;
    }
    .code-block {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 0.3rem;
        font-family: 'Courier New', Courier, monospace;
    }
    .metadata {
        font-size: 0.8rem;
        color: #666;
    }
    .code-result {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        margin-bottom: 1.5rem;
    }
    
    .code-content {
        margin: 1rem 0;
    }
    
    .code-content details {
        margin: 0.5rem 0;
    }
    
    .code-content summary {
        cursor: pointer;
        color: #007bff;
        padding: 0.5rem;
        background-color: #e9ecef;
        border-radius: 4px;
    }
    
    .code-content summary:hover {
        background-color: #dee2e6;
    }
    
    .code-block {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        overflow-x: auto;
        font-size: 0.9em;
        line-height: 1.4;
    }
    
    .matching-line {
        margin: 1rem 0;
        border-left: 3px solid #28a745;
        padding-left: 0.5rem;
    }
    
    .line-number {
        color: #6c757d;
        font-size: 0.8em;
        font-weight: bold;
        margin-bottom: 0.25rem;
    }
    
    .line-context {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 0;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        overflow-x: auto;
        font-size: 0.9em;
    }
    
    .file-info {
        color: #495057;
        font-size: 0.9em;
        margin: 0.5rem 0;
        padding: 0.5rem;
        background-color: #e9ecef;
        border-radius: 4px;
    }
    
    .similarity-score {
        color: #28a745;
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_collection' not in st.session_state:
    st.session_state.current_collection = None

def format_search_results(results: Dict[str, Any], search_type: str) -> str:
    """Format search results for display"""
    formatted_output = []
    
    # Ensure we have results to process
    if not results or not isinstance(results, dict):
        return "<div class='search-result'>No results found</div>"
    
    # Get the results list
    results_list = results.get('results', [])
    if not results_list:
        return "<div class='search-result'>No matching results found</div>"
    
    for idx, result in enumerate(results_list, 1):
        try:
            if search_type == 'code':
                # Format code search results
                formatted_output.append(f"""
<div class="search-result">
    <div class="similarity-score">Match Score: {result.get('similarity', 0):.2%}</div>
    <div class="code-block">{result.get('code', '')}</div>
    <div class="metadata">
        File: {result.get('file_info', {}).get('file_path', 'N/A')}
        Language: {result.get('file_info', {}).get('language', 'N/A')}
    </div>
    <div class="matching-lines">
        Matching Lines:
        {format_matching_lines(result.get('matched_lines', []))}
    </div>
</div>
""")
            else:
                # Format document search results
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                similarity = result.get('similarity', 0)
                relevance_factors = result.get('relevance_factors', {})
                
                # Extract page information from metadata
                page_info = f"Page {metadata.get('page_label', 'N/A')}" if metadata else "Page N/A"
                
                formatted_output.append(f"""
<div class="search-result">
    <div class="similarity-score">Relevance Score: {similarity:.2%}</div>
    <div class="content">{content}</div>
    <div class="metadata">
        {page_info} | Source: {metadata.get('source', 'N/A')}
        {format_relevance_factors(relevance_factors)}
    </div>
</div>
""")
        except Exception as e:
            logger.error(f"Error formatting result {idx}: {str(e)}")
            continue
    
    return "\n".join(formatted_output)

def format_matching_lines(lines):
    """Format matching lines with error handling"""
    if not lines:
        return "No specific line matches"
    try:
        return "<br>".join([
            f"Line {line.get('line_number', '?')}: {line.get('content', '')}" 
            for line in lines
        ])
    except Exception as e:
        logger.error(f"Error formatting matching lines: {str(e)}")
        return "Error displaying matching lines"

def format_relevance_factors(factors):
    """Format relevance factors with error handling"""
    if not factors:
        return ""
    try:
        return f"""
        <br>Semantic Similarity: {factors.get('semantic_similarity', 0):.2%}
        <br>Keyword Matches: {dict(factors.get('keyword_matches', {}))}
        <br>Context: {factors.get('context_summary', 'N/A')}
        """
    except Exception as e:
        logger.error(f"Error formatting relevance factors: {str(e)}")
        return ""

def format_code_results(results: Dict[str, Any]) -> str:
    """Format code search results for display"""
    if not results or not results.get('results'):
        return "<div class='search-result'>No matching code found</div>"
    
    formatted_output = []
    for result in results['results']:
        try:
            # Get code content
            code_content = result.get('code', '')
            
            # Format matching lines with context
            matching_lines_html = ""
            for line in result.get('matched_lines', []):
                context = line.get('context', '')
                if context:
                    matching_lines_html += f"""
                    <div class="matching-line">
                        <div class="line-number">Line {line['line_number']}</div>
                        <pre class="line-context">{context}</pre>
                    </div>
                    """
            
            # Create result card with full code view option
            formatted_output.append(f"""
            <div class="search-result code-result">
                <div class="similarity-score">Match Score: {result.get('similarity', 0):.2%}</div>
                <div class="file-info">
                    File: {result.get('file_info', {}).get('file_path', 'N/A')}
                    <br>Language: {result.get('language', 'unknown')}
                </div>
                <div class="code-content">
                    <details>
                        <summary>View Full Code</summary>
                        <pre class="code-block">{code_content}</pre>
                    </details>
                </div>
                <div class="matching-lines">
                    <h4>Matching Sections:</h4>
                    {matching_lines_html}
                </div>
            </div>
            """)
        except Exception as e:
            logger.error(f"Error formatting code result: {str(e)}")
            continue
    
    return "\n".join(formatted_output)

# Create three-column layout
left_col, main_col, right_col = st.columns([1, 2, 1])

# Left Column - Search Configuration
with left_col:
    st.header("🔍 Search Settings")
    
    # Search Type Selection with clear labels
    search_type = st.radio(
        "Select Search Type",
        ["Document", "Code"],
        help="Choose whether to search through documents or code files"
    )
    
    # Collection Selection
    try:
        collections = db_manager.client.list_collections()
        collection_names = [c.name for c in collections]
        
        if collection_names:
            # Filter collections based on search type
            if search_type == "Code":
                available_collections = [c for c in collection_names if any(ext in c.lower() for ext in ['py', 'sql', 'yml', 'yaml'])]
                if not available_collections:
                    st.warning("No code collections found. Please upload some code files first.")
                    selected_collection = None
                else:
                    selected_collection = st.selectbox(
                        "Select Code Collection",
                        available_collections,
                        help="Choose which collection of code files to search through"
                    )
            else:
                available_collections = [c for c in collection_names if 'pdf' in c.lower()]
                if not available_collections:
                    st.warning("No document collections found. Please upload some documents first.")
                    selected_collection = None
                else:
                    selected_collection = st.selectbox(
                        "Select Document Collection",
                        available_collections,
                        help="Choose which collection of documents to search through"
                    )
        else:
            st.warning("No collections available. Please upload some files first.")
            selected_collection = None
            
    except Exception as e:
        st.error(f"Error loading collections: {str(e)}")
        selected_collection = None

    # Code-specific options
    if search_type == "Code":
        st.subheader("Code Search Options")
        
        # Language selection
        language = st.selectbox(
            "Programming Language",
            ["python", "sql", "yaml"],
            help="Select the programming language to search in"
        )
        
        # Search mode
        search_mode = st.radio(
            "Search Mode",
            ["Semantic", "Pattern"],
            help="Semantic: Find similar code concepts\nPattern: Find exact matches"
        )

    # Common search settings
    st.subheader("Search Settings")
    n_results = st.slider(
        "Number of Results",
        min_value=1,
        max_value=10,
        value=3,
        help="How many results to return"
    )

# Main Column - Chat Interface
with main_col:
    st.header("💬 Search Chat")
    
    # Add clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.container():
            st.markdown(f"""
            <div class="chat-message {message['role']}">
                <div class="message">{message['content']}</div>
                {message.get('results_html', '')}
            </div>
            """, unsafe_allow_html=True)
    
    # Query input
    query = st.text_area("Enter your search query:", key="query")
    if st.button("Search") and query:
        if not selected_collection:
            if search_type == "Code":
                st.error("Please upload code files first.")
            else:
                st.error("Please upload documents first.")
        else:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": query
            })
            
            try:
                if search_type == "Code":
                    results = search_processor.search_code(
                        code_query=query,
                        collection_name=selected_collection,
                        language=language,
                        n_results=n_results
                    )
                    results_html = format_code_results(results)
                else:
                    results = search_processor.search_documents(
                        query=query,
                        collection_name=selected_collection,
                        n_results=n_results
                    )
                    results_html = format_search_results(results, 'document')
                
                # Add assistant response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Here are the most relevant results:",
                    "results_html": results_html
                })
                
                # Rerun to update display
                st.rerun()
            
            except Exception as e:
                st.error(f"Error processing search: {str(e)}")

# Right Column - File Upload
with right_col:
    st.header("📁 Upload Files")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload Document or Code",
        type=['pdf', 'py', 'sql', 'yml', 'yaml'],
        key="file_uploader"
    )
    
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if st.button(f"Process {file_type.upper()} File"):
            try:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Save uploaded file
                status_text.text("Saving uploaded file...")
                upload_dir = BASE_DIR / "uploads"
                upload_dir.mkdir(exist_ok=True)
                file_path = upload_dir / uploaded_file.name
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                progress_bar.progress(25)
                status_text.text("Processing file...")
                
                # Process file based on type
                if file_type == 'pdf':
                    docs = db_manager.process_document(str(file_path), 'pdf')
                elif file_type in ['py', 'sql', 'yml', 'yaml']:
                    docs = db_manager.process_code(str(file_path), file_type)
                
                if not docs:
                    raise ValueError("No content extracted from file")
                
                progress_bar.progress(50)
                status_text.text("Adding to collection...")
                
                # Add to appropriate collection
                collection_name = f"{file_type}_documents"
                db_manager.add_documents(
                    collection_name,
                    docs,
                    metadata={
                        "source": file_type,
                        "file_path": str(file_path),
                        "filename": uploaded_file.name
                    }
                )
                
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                st.success(f"""
                Successfully processed {uploaded_file.name}:
                - Created {len(docs)} chunks
                - Added to collection: {collection_name}
                """)
                
            except Exception as e:
                st.error(f"""
                Error processing file: {str(e)}
                
                Please make sure:
                1. The file is not corrupted
                2. The file is properly formatted
                3. You have sufficient permissions
                """)
                logger.error(f"File processing error: {str(e)}", exc_info=True)
            
            finally:
                # Cleanup
                if 'progress_bar' in locals():
                    progress_bar.empty()
                if 'status_text' in locals():
                    status_text.empty()

    # Display currently available collections
    st.subheader("Available Collections")
    try:
        collections = db_manager.client.list_collections()
        if collections:
            for collection in collections:
                with st.expander(f"📚 {collection.name}"):
                    try:
                        count = len(collection.get()['ids'])
                        st.text(f"Documents: {count}")
                    except:
                        st.text("Unable to get document count")
        else:
            st.info("No collections available yet. Upload some files!")
    except Exception as e:
        st.warning(f"Unable to list collections: {str(e)}") 