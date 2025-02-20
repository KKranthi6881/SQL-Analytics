import streamlit as st
import requests
import json
from pathlib import Path
import os
from datetime import datetime
from db.database import ChatDatabase
import plotly.graph_objects as go
import pandas as pd
from src.app import app
from src.agents.sql_analysis_system import SQLAnalysisSystem
from typing import Optional, Dict, Any

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
API_URL = "http://localhost:8000"  # Adjust port if needed

def init_session_state():
    if 'selected_conversation' not in st.session_state:
        st.session_state.selected_conversation = None

def render_history_sidebar():
    """Render the conversation history in the sidebar"""
    with st.sidebar:
        # Simple header
        st.title("History")
        
        # Custom styling for buttons and sidebar
        st.markdown("""
            <style>
            /* Sidebar styling */
            [data-testid="stSidebar"] {
                background-color: #ffffff;
                border-right: 1px solid #e2e8f0;
            }
            
            /* Button styling */
            [data-testid="stButton"] button {
                background-color: #f8fafc !important;
                color: #1e293b !important;
                border: 1px solid #e2e8f0 !important;
            }
            [data-testid="stButton"] button:hover {
                background-color: #f1f5f9 !important;
                border-color: #cbd5e0 !important;
            }
            
            /* History item styling */
            .element-container button {
                text-align: left !important;
                padding: 0.75rem !important;
                margin: 0.25rem 0 !important;
                border-radius: 4px !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        if st.button("+ New Analysis", type="primary", use_container_width=True):
            st.session_state.selected_conversation = None
            st.rerun()
        
        st.divider()
        
        # Search box
        search_term = st.text_input("", 
                                  placeholder="🔍 Search conversations...",
                                  label_visibility="collapsed")
        
        # Get history from database
        db = ChatDatabase()
        history = db.get_conversation_history_with_checkpoints()
        
        if not history:
            st.info("No conversations yet. Start a new chat!")
            return

        # Sort conversations by date
        sorted_history = sorted(history, key=lambda x: x['timestamp'], reverse=True)
        
        # Display conversations
        for conv in sorted_history:
            if search_term.lower() in conv['query'].lower():
                # Format timestamp
                timestamp = datetime.strptime(conv['timestamp'].split('.')[0], '%Y-%m-%d %H:%M:%S')
                formatted_time = timestamp.strftime('%b %d, %Y %I:%M %p')
                
                # Clickable conversation item
                if st.sidebar.button(
                    f"{conv['query'][:50]}...\n📅 {formatted_time}", 
                    key=f"conv_{conv['id']}", 
                    use_container_width=True,
                    type="secondary"
                ):
                    st.session_state.selected_conversation = conv
                    st.rerun()

def save_checkpoint(conversation_id: str, original_response: dict, updated_response: dict):
    """Save a checkpoint when a conversation response is edited"""
    try:
        db = ChatDatabase()
        checkpoint_data = {
            'original_response': original_response,
            'updated_response': updated_response,
            'edit_timestamp': datetime.now().isoformat()
        }
        
        # Create a checkpoint in the checkpoints table
        db.create_checkpoint(
            thread_id=conversation_id,
            checkpoint_id=f"edit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            checkpoint_type="response_edit",
            checkpoint_data=json.dumps(checkpoint_data)
        )
        return True
    except Exception as e:
        st.error(f"Error saving checkpoint: {str(e)}")
        return False

def display_conversation_details(conversation):
    """Display the selected conversation details with response edit functionality"""
    if conversation:
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.markdown("### Conversation Details")
        with col2:
            if st.button("← Back"):
                st.session_state.selected_conversation = None
                st.rerun()
        
        # Display timestamp
        st.markdown(f"*{conversation['timestamp']}*")
        
        # Display query (read-only)
        st.markdown("#### 🔍 Query")
        st.code(conversation['query'], language='text')
        
        # Display response with edit functionality
        st.markdown("#### 💡 Response")
        
        # Initialize session state for editing
        if 'editing_response' not in st.session_state:
            st.session_state.editing_response = False
            st.session_state.edited_response = None
            st.session_state.original_format = None
            st.session_state.original_structure = None

        # Edit/Cancel and Save buttons in the same row
        if st.session_state.editing_response:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.button("❌ Cancel", key="cancel_edit_btn", on_click=lambda: handle_cancel())
            with col2:
                save_button = st.button("💾 Save Changes", key="save_response_btn", type="primary", use_container_width=True)
        else:
            if st.button("✏️ Edit", key="edit_response_btn"):
                st.session_state.editing_response = True
                output = conversation['output']
                st.session_state.original_structure = output
                if isinstance(output, dict):
                    st.session_state.edited_response = json.dumps(output, indent=2)
                    st.session_state.original_format = 'json'
                else:
                    st.session_state.edited_response = str(output)
                    st.session_state.original_format = 'text'
                st.rerun()

        if st.session_state.editing_response:
            # Show side-by-side edit and preview
            col1, col2 = st.columns([0.5, 0.5])
            
            with col1:
                st.markdown("##### Edit Response")
                edited_response = st.text_area(
                    "",  # Empty label
                    value=st.session_state.edited_response,
                    height=400,
                    key="response_editor",
                    label_visibility="collapsed"
                )
            
            with col2:
                st.markdown("""
                    <style>
                        /* Container styling */
                        .preview-section {
                            margin-top: 0;
                            padding: 0;
                        }
                        .preview-header {
                            margin-bottom: 8px;
                            font-size: 1rem;
                            font-weight: 600;
                        }
                        .preview-container {
                            border: 1px solid #e2e8f0;
                            border-radius: 4px;
                            padding: 10px;
                            background-color: #ffffff;
                            height: 400px;
                            overflow-y: auto;
                            font-family: monospace;
                            font-size: 0.875rem;
                            line-height: 1.5;
                        }
                        
                        /* Remove extra spacing */
                        .preview-container > div:first-child {
                            margin-top: 0 !important;
                        }
                        .preview-container > div:last-child {
                            margin-bottom: 0 !important;
                        }
                        
                        /* Code block styling */
                        .preview-container pre {
                            margin: 0;
                            padding: 0;
                            background: transparent;
                        }
                        .preview-container code {
                            white-space: pre-wrap;
                            word-break: break-word;
                        }
                        
                        /* Markdown content styling */
                        .preview-container .stMarkdown {
                            margin: 0;
                            padding: 0;
                        }
                        
                        /* Error message styling */
                        .preview-container .stAlert {
                            margin: 0;
                            padding: 8px;
                        }
                    </style>
                    <div class="preview-section">
                        <div class="preview-header">Live Preview</div>
                    </div>
                """, unsafe_allow_html=True)
                
                with st.container():
                    st.markdown('<div class="preview-container">', unsafe_allow_html=True)
                    try:
                        if st.session_state.original_format == 'json':
                            # Try to parse and display as JSON
                            parsed_json = json.loads(edited_response)
                            st.code(
                                json.dumps(parsed_json, indent=2),
                                language='json'
                            )
                        else:
                            st.markdown(edited_response)
                    except json.JSONDecodeError:
                        st.error("⚠️ Invalid JSON format")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Handle save action
            if save_button:
                try:
                    if st.session_state.original_format == 'json':
                        new_response = json.loads(edited_response)
                        formatted_response = json.dumps(new_response)
                    else:
                        new_response = edited_response
                        formatted_response = edited_response
                    
                    if new_response != st.session_state.original_structure:
                        if save_checkpoint(conversation['id'], st.session_state.original_structure, new_response):
                            db = ChatDatabase()
                            db.update_conversation_response(conversation['id'], formatted_response)
                            
                            # Update session state
                            st.session_state.editing_response = False
                            st.session_state.edited_response = None
                            st.session_state.original_format = None
                            st.session_state.original_structure = None
                            st.session_state.selected_conversation['output'] = new_response
                            st.success("✅ Response updated successfully!")
                            st.rerun()
                    else:
                        st.warning("ℹ️ No changes detected")
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON format. Please check your edits.")
                    st.markdown("""
                        **JSON Format Tips:**
                        - Use double quotes `"` for strings
                        - Quote all property names
                        - No trailing commas
                        - Proper nesting of brackets and braces
                    """)
        else:
            # Show readonly response in original format
            output = conversation['output']
            if isinstance(output, dict):
                st.json(output)
            else:
                st.markdown(output)
        
        # Display Vector Search if available
        if conversation['code_context']:
            st.markdown("#### 📝 Vector Search Context")
            st.json(conversation['code_context'])
            
        # Display technical details if available
        if conversation['technical_details']:
            with st.expander("🔧 Technical Details"):
                st.json(conversation['technical_details'])
                
        # Display edit history
        if conversation['checkpoints']:
            with st.expander("🔄 Edit History"):
                for checkpoint in conversation['checkpoints']:
                    if checkpoint['type'] == 'response_edit':
                        checkpoint_data = checkpoint['checkpoint']
                        st.markdown(f"""
                            **Edit made on:** {datetime.fromisoformat(checkpoint_data['edit_timestamp']).strftime('%Y-%m-%d %I:%M %p')}
                            
                            **Original Response:**
                        """)
                        if isinstance(checkpoint_data['original_response'], dict):
                            st.json(checkpoint_data['original_response'])
                        else:
                            st.code(checkpoint_data['original_response'])
                            
                        st.markdown("**Updated Response:**")
                        if isinstance(checkpoint_data['updated_response'], dict):
                            st.json(checkpoint_data['updated_response'])
                        else:
                            st.code(checkpoint_data['updated_response'])
                        
                        st.divider()

def handle_cancel():
    """Handle cancel button click"""
    st.session_state.editing_response = False
    st.session_state.edited_response = None
    st.session_state.original_format = None
    st.session_state.original_structure = None
    st.rerun()

def analyze_sql_query(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Send SQL analysis request to the API"""
    try:
        # Create initial state for the request
        request_data = {
            "query": query,
            "context": context
        }
        
        response = requests.post(
            f"{API_URL}/sql_analyze/",
            json=request_data
        )
        response.raise_for_status()
        
        result = response.json()
        if "error" in result:
            st.error(f"Analysis error: {result['error']}")
            return {"error": result["error"]}
            
        return result
        
    except requests.exceptions.RequestException as e:
        st.error(f"SQL Analysis failed: {str(e)}")
        return {"error": str(e)}

def get_database_schema() -> Optional[Dict]:
    """Get database schema from the API"""
    try:
        response = requests.get(f"{API_URL}/database_schema/")
        response.raise_for_status()
        return response.json().get("schema")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch schema: {str(e)}")
        return None

def analyze_code(query: str) -> Dict[str, Any]:
    """Send code analysis request to the API"""
    try:
        response = requests.post(
            f"{API_URL}/analyze/",
            json={"query": query}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Code analysis failed: {str(e)}")
        return {"error": str(e)}

def upload_file(file) -> Optional[Dict]:
    """Upload file to the API"""
    try:
        files = {"file": file}
        response = requests.post(f"{API_URL}/upload/", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"File upload failed: {str(e)}")
        return None

def main():
    init_session_state()
    
    # Render sidebar
    render_history_sidebar()
    
    # Main content area
    if st.session_state.selected_conversation:
        display_conversation_details(st.session_state.selected_conversation)
        
        if st.button("Start New Analysis", key="new_analysis_main"):
            st.session_state.selected_conversation = None
            st.rerun()
    else:
        st.title("SQL Analytics Assistant")
        
        # Create tabs for different types of analysis
        tab1, tab2, tab3, tab4 = st.tabs(["SQL Analysis", "Code Analysis", "File Upload", "History"])
        
        with tab1:
            st.markdown("### SQL Query Analysis")
            
            # Add schema viewer
            if st.checkbox("View Database Schema", key="view_schema"):
                try:
                    schema = requests.get(f"{API_URL}/database_schema/").json()
                    if schema and "schema" in schema:
                        st.success(f"Found {len(schema['schema'])} tables")
                        for table, details in schema['schema'].items():
                            with st.expander(f"📊 {table}"):
                                df = pd.DataFrame(details)
                                st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading schema: {str(e)}")
            
            # Query input
            query = st.text_area(
                "Enter your SQL question:",
                placeholder="e.g., What are the top 10 movies by revenue?",
                height=100
            )
            
            if st.button("Analyze Query", type="primary"):
                if query:
                    with st.spinner("Analyzing..."):
                        result = analyze_sql_query(query)
                        
                        if "error" in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            # Display Analysis
                            with st.expander("Analysis Details", expanded=True):
                                st.json(result["result"].get("analysis", {}))
                            
                            # Display SQL Query
                            with st.expander("Generated SQL", expanded=True):
                                query_info = result["result"].get("query", {})
                                st.code(query_info.get("query", "No query available"), language="sql")
                                
                                if hints := query_info.get("performance_hints"):
                                    st.info("Performance Hints:")
                                    for hint in hints:
                                        st.write(f"• {hint}")
                            
                            # Display Results
                            if data := result["result"].get("results"):
                                st.subheader("Query Results")
                                df = pd.DataFrame(data)
                                st.dataframe(df)
                                st.info(f"Found {len(df)} rows")
                            
                            # Display Visualization
                            if viz := result["result"].get("visualization"):
                                st.subheader("Visualization")
                                try:
                                    exec(viz["plotly_code"])
                                    fig = locals()["fig"]
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Visualization error: {str(e)}")
                else:
                    st.warning("Please enter a question to analyze.")
        
        with tab2:
            # Code Analysis Tab
            st.markdown("### Code Analysis")
            code_analysis_interface()
        
        with tab3:
            # File Upload Tab
            st.markdown("### File Upload")
            file_upload_interface()

def get_analyst():
    """Get the DataAnalysisSystem instance"""
    return SQLAnalysisSystem()

def code_analysis_interface():
    query = st.text_area("Enter your code analysis query:", height=100)
    
    if st.button("Analyze Code"):
        if not query:
            st.warning("Please enter a query first.")
            return
            
        with st.spinner("Analyzing..."):
            result = analyze_code(query)
            
            if result:
                st.markdown("#### Analysis Result")
                st.markdown(result.get("output", "No output available"))
                
                if result.get("code_context"):
                    st.markdown("#### Vector Search Context")
                    st.json(result["code_context"])

def file_upload_interface():
    uploaded_file = st.file_uploader(
        label="Choose a file to upload",
        type=["py", "sql", "pdf"],
        key="file_upload"
    )
    
    if uploaded_file and st.button("Process Selected File"):
        with st.spinner("Processing file..."):
            result = upload_file(uploaded_file)
            if result:
                st.success(f"File processed: {result.get('filename')}")
                st.json(result)

if __name__ == "__main__":
    main() 