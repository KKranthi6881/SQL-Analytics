import streamlit as st
import requests
import json
from pathlib import Path
import os
from datetime import datetime
from db.database import ChatDatabase

# Set page config with modern styling
st.set_page_config(
    page_title="Code Research Assistant",
    page_icon="🔍",
    layout="wide"
)

# Modern UI styling
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Main Content Area */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        padding: 0;
    }
    
    /* Sidebar Header */
    .sidebar-header {
        background: linear-gradient(135deg, #4f46e5, #3b82f6);
        color: white;
        padding: 1.5rem;
        margin: -1rem -1rem 1rem -1rem;
        border-radius: 0 0 12px 12px;
    }
    
    /* Conversation Items */
    .conversation-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.75rem 0;
        transition: all 0.2s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        cursor: pointer;
    }
    
    .conversation-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-color: #3b82f6;
    }
    
    /* Query Preview */
    .query-preview {
        color: #1e293b;
        font-weight: 500;
        font-size: 0.95rem;
        line-height: 1.4;
        margin-bottom: 0.5rem;
    }
    
    /* Timestamp and Badges */
    .timestamp {
        color: #64748b;
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .checkpoint-badge {
        background: #f1f5f9;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        color: #64748b;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    /* Date Headers */
    .date-header {
        background: #f8fafc;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-weight: 500;
        color: #475569;
        font-size: 0.875rem;
    }
    
    /* Buttons */
    .stButton button {
        background: #3b82f6;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background: #2563eb;
        transform: translateY(-1px);
    }
    
    /* Search Box */
    .search-container {
        background: #f1f5f9;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .stTextInput input {
        border: 2px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.5rem;
        font-size: 0.95rem;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 2px dashed #e2e8f0;
        margin: 1rem 0;
    }
    
    /* Code Blocks and JSON */
    .stCodeBlock, div.json-output {
        background: #ffffff;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e0;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
API_URL = "http://localhost:8000"  # Adjust if your FastAPI server runs on a different port

def init_session_state():
    if 'selected_conversation' not in st.session_state:
        st.session_state.selected_conversation = None

def render_history_sidebar():
    with st.sidebar:
        # Modern sidebar header
        st.markdown("""
            <div class="sidebar-header">
                <h2 style="margin:0; font-size: 1.5rem;">
                    🔍 Code Research Assistant
                </h2>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">
                    Your AI-powered code companion
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # New chat button with modern styling
        if st.button("+ New Analysis", key="new_chat_btn", use_container_width=True):
            st.session_state.selected_conversation = None
            st.rerun()
        
        # Modern search box
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        search_term = st.text_input("", 
                                  placeholder="🔍 Search conversations...",
                                  label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        # Get history from database
        db = ChatDatabase()
        history = db.get_conversation_history_with_checkpoints()
        
        if not history:
            st.info("No conversations yet. Start a new chat!")
            return

        def get_date(timestamp):
            """Parse timestamp string to date object"""
            # Remove any milliseconds/microseconds if present
            timestamp = timestamp.split('.')[0]
            return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').date()

        def format_timestamp(ts):
            """Format timestamp for display"""
            # Remove any milliseconds/microseconds if present
            ts = ts.split('.')[0]
            dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
            return dt.strftime('%I:%M %p')

        # Sort and group conversations
        sorted_history = sorted(history, key=lambda x: x['timestamp'], reverse=True)
        grouped_history = {}
        for conv in sorted_history:
            date = get_date(conv['timestamp'])
            if date not in grouped_history:
                grouped_history[date] = []
            grouped_history[date].append(conv)

        # Display grouped conversations
        for date in sorted(grouped_history.keys(), reverse=True):
            st.markdown(f"""
                <div class="date-header">
                    📅 {date.strftime("%B %d, %Y")}
                </div>
            """, unsafe_allow_html=True)
            
            for conv in grouped_history[date]:
                if search_term.lower() in conv['query'].lower():
                    col1, col2 = st.columns([0.85, 0.15])
                    
                    with col1:
                        st.markdown(f"""
                            <div class="conversation-item">
                                <div class="query-preview">{conv['query'].split('\n')[0][:50]}...</div>
                                <div class="timestamp">
                                    <span>🕒 {format_timestamp(conv['timestamp'])}</span>
                                    <span class="checkpoint-badge">
                                        {len(conv.get('checkpoints', [])) if conv.get('checkpoints') else 0} checkpoints
                                    </span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if st.button("View", key=f"btn_{conv['id']}", use_container_width=True):
                            st.session_state.selected_conversation = conv
                            st.rerun()

def display_conversation_details(conversation):
    """Display the selected conversation details"""
    if conversation:
        # Add a back button
        if st.button("← Back to Chat"):
            st.session_state.selected_conversation = None
            st.rerun()

        st.markdown("### Conversation Details")
        
        # Display timestamp
        st.markdown(f"*{conversation['timestamp']}*")
        
        # Display query
        st.markdown("#### 🔍 Query")
        st.code(conversation['query'], language='text')
        
        # Display response
        st.markdown("#### 💡 Response")
        if isinstance(conversation['output'], dict):
            st.json(conversation['output'])
        else:
            st.write(conversation['output'])
        
        # Display code context if available
        if conversation['code_context']:
            st.markdown("#### 📝 Code Context")
            st.json(conversation['code_context'])
            
        # Display technical details if available
        if conversation['technical_details']:
            with st.expander("🔧 Technical Details"):
                st.json(conversation['technical_details'])
                
        # Display checkpoints if available
        if conversation['checkpoints']:
            with st.expander("🔄 Conversation Checkpoints"):
                for checkpoint in conversation['checkpoints']:
                    st.markdown(f"**Checkpoint: {checkpoint['checkpoint_id']}**")
                    if checkpoint['parent_id']:
                        st.markdown(f"*Parent: {checkpoint['parent_id']}*")
                    st.markdown(f"Type: {checkpoint['type']}")
                    
                    # Display checkpoint data in a collapsible section
                    with st.expander("View Checkpoint Data"):
                        st.json(checkpoint['checkpoint'])
                    
                    # Display metadata if it exists
                    if checkpoint['metadata']:
                        with st.expander("View Metadata"):
                            st.json(checkpoint['metadata'])
                    
                    st.divider()

def analyze_code(query: str):
    """Send analysis request to the backend"""
    try:
        response = requests.post(
            f"{API_URL}/analyze/",
            json={"query": query}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with server: {str(e)}")
        return None

def upload_file(file):
    """Upload file to the backend"""
    try:
        files = {"file": file}
        response = requests.post(f"{API_URL}/upload/", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading file: {str(e)}")
        return None

def main():
    init_session_state()
    
    # Render sidebar
    render_history_sidebar()
    
    # Main content area
    if st.session_state.selected_conversation:
        display_conversation_details(st.session_state.selected_conversation)
        
        # Add "Start New Chat" button in main area
        if st.button("Start New Chat", key="new_chat_main"):
            st.session_state.selected_conversation = None
            st.rerun()
    else:
        st.title("Code Analysis System")
        st.markdown("### Upload and analyze your code")

    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a file to upload (.py, .sql, .pdf)", 
        type=["py", "sql", "pdf"]
    )
    
    if uploaded_file:
        if st.button("Process File"):
            with st.spinner("Processing file..."):
                result = upload_file(uploaded_file)
                if result:
                    st.success(f"File processed: {result.get('filename')}")
                    st.json(result)

    # Analysis section
    st.markdown("### Code Analysis")
    query = st.text_area("Enter your analysis query:", height=100)
    
    if st.button("Analyze"):
        if not query:
            st.warning("Please enter a query first.")
            return
            
        with st.spinner("Analyzing..."):
            result = analyze_code(query)
            
            if result:
                # Display the analysis output
                st.markdown("#### Analysis Result")
                st.markdown(result.get("output", "No output available"))
                
                # Display code context if available
                code_context = result.get("code_context", {})
                if code_context:
                    st.markdown("#### Code Context")
                    st.json(code_context)

if __name__ == "__main__":
    main() 