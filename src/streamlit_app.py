import streamlit as st
import sys
from pathlib import Path
import os

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Now import local modules
from src.utils import ChromaDBManager
from src.processor import SearchProcessor
import yaml
import logging
from typing import Dict, Any
from src.agents.code_search import AnalysisSystem
from src.tools import SearchTools

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

def display_analysis_results(results):
    """Display the analysis results in a structured way"""
    st.write("### Analysis Results")
    
    # Display summary
    st.write("#### Summary")
    st.write(results.get("summary", "No summary available"))
    
    # Display code context if available
    if code_context := results.get("code_context", {}):
        st.write("#### Code Analysis")
        with st.expander("View Code Details"):
            if sql_results := code_context.get("results", {}).get("sql", []):
                st.write("SQL Results:")
                for idx, result in enumerate(sql_results, 1):
                    st.code(result.get("code", ""), language="sql")
                    
            if py_results := code_context.get("results", {}).get("python", []):
                st.write("Python Results:")
                for idx, result in enumerate(py_results, 1):
                    st.code(result.get("code", ""), language="python")
    
    # Display documentation context if available
    if doc_context := results.get("doc_context", {}):
        st.write("#### Documentation")
        with st.expander("View Documentation Details"):
            for idx, result in enumerate(doc_context.get("results", []), 1):
                st.write(f"Document {idx}:")
                st.write(result.get("content", ""))
                st.write("---")

# Update the main layout to use tabs instead
tab1, tab2, tab3 = st.tabs(["💬 Code Chat", "🔍 Document Search", "📁 File Management"])

# Tab 1: Code Chat
with tab1:
    st.header("Code Analysis Chat")
    st.write("Ask questions about your code and get context-aware answers.")

    def initialize_analysis_system():
        """Initialize the Analysis System."""
        if not hasattr(st.session_state, 'analysis_system'):
            try:
                tools = SearchTools(db_manager)
                st.session_state.analysis_system = AnalysisSystem(tools)
                logger.info("Analysis System initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Analysis System: {str(e)}")
                st.error(f"Error initializing Analysis System: {str(e)}")
                return False
        return True

    # Initialize analysis system
    if initialize_analysis_system():
        # Chat interface
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant":
                    # Show code context
                    if msg.get("code_context"):
                        with st.expander("View Code Analysis"):
                            st.code(msg["code_context"])
                    
                    # Show documentation context
                    if msg.get("doc_context"):
                        with st.expander("View Documentation Analysis"):
                            st.markdown(msg["doc_context"])

        # Chat input
        if query := st.chat_input("Ask about your code..."):
            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.write(query)

            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing code and documentation..."):
                    try:
                        results = st.session_state.analysis_system.analyze(query)
                        display_analysis_results(results)
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": results["summary"],
                            "code_context": results.get("code_context"),
                            "doc_context": results.get("doc_context")
                        })
                    
                    except Exception as e:
                        error_msg = f"Error analyzing query: {str(e)}"
                        logger.error(error_msg)
                        st.error(error_msg)

# Tab 2: Document Search
with tab2:
    st.header("Document Search")
    st.write("Search through your uploaded documents with semantic search.")
    
    # Your existing document search code here...
    # (Keep the document search functionality from your current implementation)

# Tab 3: File Management
with tab3:
    st.header("File Management")
    
    # File upload section
    st.subheader("Upload Files")
    uploaded_file = st.file_uploader(
        "Upload Code or Documents",
        type=['pdf', 'py', 'sql', 'yml', 'yaml'],
        help="Upload your code files or documents for analysis"
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

    # Collection Management
    st.subheader("Manage Collections")
    try:
        collections = db_manager.client.list_collections()
        if collections:
            for collection in collections:
                with st.expander(f"📚 {collection.name}"):
                    count = len(collection.get()['ids'])
                    st.text(f"Documents: {count}")
                    if st.button(f"Delete {collection.name}", key=f"del_{collection.name}"):
                        # Add collection deletion functionality
                        pass
        else:
            st.info("No collections available. Upload some files to get started!")
    except Exception as e:
        st.error(f"Error loading collections: {str(e)}") 