from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from .utils import ChromaDBManager
from .tools import SearchTools
from .agents.code_research import SimpleAnalysisSystem
import logging
from pydantic import BaseModel
from .db.database import ChatDatabase
import streamlit as st
from datetime import datetime
from src.agents.sql_analysis_system import SQLAnalysisSystem, AnalysisOutput, AnalysisState
from langchain_community.utilities import SQLDatabase
from src.tools import SearchTools
from typing import Optional, Dict, Any
import time
from langchain.schema import HumanMessage

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Create required directories
for directory in ["static", "templates", "uploads"]:
    (BASE_DIR / directory).mkdir(exist_ok=True)

# Create a templates directory and mount it
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Initialize ChromaDB manager
db_manager = ChromaDBManager(persist_directory=str(BASE_DIR / "chroma_db"))

# Initialize tools and analysis system
code_search_tools = SearchTools(db_manager)
analysis_system = SimpleAnalysisSystem(code_search_tools)

# Initialize database
chat_db = ChatDatabase()

# Initialize SQL Analysis System
sql_analysis_system = SQLAnalysisSystem(str(Path(__file__).parent / "db" / "sampledb" / "sakila_master.db"))

# Add new model for code analysis requests
class CodeAnalysisRequest(BaseModel):
    query: str

# Add new model for analysis requests
class AnalysisRequest(BaseModel):
    query: str

# Add new request/response models
class SQLAnalysisRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class SQLAnalysisResponse(BaseModel):
    result: Dict[str, Any]
    execution_time: float

# Add new endpoints for code analysis
@app.post("/analyze/")
async def analyze_code(request: CodeAnalysisRequest):
    """
    Endpoint for code analysis using the new agent system
    """
    try:
        logger.info(f"Analyzing query: {request.query}")
        
        # Use the new analysis system
        result = analysis_system.analyze(request.query)
        
        return JSONResponse({
            "status": "success",
            "output": result.get("output", "No output available"),
            "code_context": result.get("code_context", {})
        })
        
    except Exception as e:
        logger.error(f"Error in code analysis: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Log the upload attempt
        logger.info(f"Attempting to upload file: {file.filename}")
        
        # Create uploads directory if it doesn't exist
        upload_dir = BASE_DIR / "uploads"
        upload_dir.mkdir(exist_ok=True)
        
        # Save the uploaded file
        file_path = upload_dir / file.filename
        logger.info(f"Saving file to: {file_path}")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the file based on its extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        collection_name = f"{file_extension[1:]}_documents"
        
        logger.info(f"Processing file with extension: {file_extension}")
        
        if file_extension == '.pdf':
            logger.info("Processing PDF file...")
            docs = db_manager.process_document(str(file_path), 'pdf')
        elif file_extension == '.sql':
            logger.info("Processing SQL file...")
            docs = db_manager.process_code(str(file_path), 'sql')
        elif file_extension == '.py':
            logger.info("Processing Python file...")
            docs = db_manager.process_code(str(file_path), 'python')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        logger.info(f"Adding {len(docs)} documents to collection: {collection_name}")
        
        # Add documents to collection
        db_manager.add_documents(
            collection_name,
            docs,
            metadata={"source": file_extension[1:], "file_path": str(file_path)}
        )
        
        return {
            "filename": file.filename,
            "status": "File processed successfully",
            "collection": collection_name,
            "num_documents": len(docs)
        }
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query_collection(collection_name: str, query_text: str):
    try:
        results = db_manager.query_collection(collection_name, query_text)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}

# Add endpoint to get available collections
@app.get("/collections/")
async def list_collections():
    """
    Get list of available collections
    """
    try:
        collections = db_manager.client.list_collections()
        return {
            "status": "success",
            "collections": [
                {
                    "name": collection.name,
                    "count": len(collection.get()['ids'])
                }
                for collection in collections
            ]
        }
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

# Add endpoint for detailed collection info
@app.get("/collections/{collection_name}/info")
async def get_collection_info(collection_name: str):
    """
    Get detailed information about a specific collection
    """
    try:
        collection = db_manager.get_or_create_collection(collection_name)
        collection_data = collection.get()
        
        return {
            "status": "success",
            "info": {
                "name": collection_name,
                "count": len(collection_data['ids']),
                "metadata": collection_data.get('metadatas', []),
                "documents": len(collection_data.get('documents', [])),
            }
        }
    except Exception as e:
        logger.error(f"Error getting collection info: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

# Add these test endpoints
@app.get("/test-db/")
async def test_db():
    """Test endpoint to check ChromaDB collections and contents"""
    try:
        collections = db_manager.client.list_collections()
        results = {}
        for collection in collections:
            data = collection.get()
            results[collection.name] = {
                "count": len(data['ids']),
                "sample": data['documents'][:2] if data['documents'] else [],
                "metadata": data.get('metadatas', [])[:2] if data.get('metadatas') else []
            }
        return JSONResponse({
            "status": "success",
            "collections": results
        })
    except Exception as e:
        logger.error(f"Error testing DB: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/test-tools/")
async def test_tools():
    """Test endpoint to check search tools functionality"""
    try:
        tools = SearchTools(db_manager)
        test_query = "workflow analysis"
        
        # Test each search function directly
        results = {
            "code_search": tools.search_code(test_query),
            "doc_search": tools.search_documentation(test_query),
            "relationship_search": tools.search_relationships(test_query)
        }
        
        return JSONResponse({
            "status": "success",
            "results": results
        })
    except Exception as e:
        logger.error(f"Error testing tools: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# Add new endpoints
@app.get("/conversations/")
async def get_conversations():
    """Get recent conversations"""
    try:
        conversations = chat_db.get_recent_conversations()
        return JSONResponse({
            "status": "success",
            "conversations": conversations
        })
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation"""
    try:
        conversation = chat_db.get_conversation(conversation_id)
        if conversation:
            return JSONResponse({
                "status": "success",
                "conversation": conversation
            })
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "Conversation not found"}
        )
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/history/")
async def get_history():
    """Get conversation history"""
    try:
        history = chat_db.get_conversation_history()
        return JSONResponse({
            "status": "success",
            "history": history
        })
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

def render_history_sidebar():
    """Render the conversation history in the sidebar"""
    with st.sidebar:
        st.header("Conversation History")
        
        # Get history from database
        history = chat_db.get_conversation_history()
        
        if not history:
            st.info("No conversation history yet")
            return
            
        for item in history:
            # Create an expander for each conversation
            with st.expander(f"Query: {item['query'][:50]}..."):
                st.text(f"Time: {item['timestamp']}")
                
                # Show the query
                st.markdown("**Query:**")
                st.text(item['query'])
                
                # Show the response
                st.markdown("**Response:**")
                st.json(item['output'])
                
                # Show code context if available
                if item['code_context']:
                    st.markdown("**Code Context:**")
                    st.json(item['code_context'])
                
                # Add a divider between conversations
                st.divider()

def chat_interface():
    """Chat interface for the Streamlit app"""
    st.markdown("### Ask a Question")
    query = st.text_area("Enter your question:", height=100)
    
    if st.button("Submit"):
        if not query:
            st.warning("Please enter a question first.")
            return
            
        with st.spinner("Analyzing..."):
            result = analyze_code(query)
            
            if result:
                st.markdown("#### Response")
                st.markdown(result.get("output", "No output available"))
                
                if result.get("code_context"):
                    st.markdown("#### Code Context")
                    st.json(result["code_context"])

def file_upload_interface():
    """File upload interface for the Streamlit app"""
    st.markdown("### Upload Files")
    uploaded_file = st.file_uploader(
        "Choose a file to upload (.py, .sql, .pdf)", 
        type=["py", "sql", "pdf"]
    )
    
    if uploaded_file and st.button("Process File"):
        with st.spinner("Processing file..."):
            result = upload_file(uploaded_file)
            if result:
                st.success(f"File processed: {result.get('filename')}")
                st.json(result)

def main():
    st.set_page_config(page_title="Code Analysis Assistant", layout="wide")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stSidebar {
            background-color: #f5f5f5;
        }
        .stExpander {
            background-color: white;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Render the history sidebar
    render_history_sidebar()
    
    # Main content
    st.title("Code Analysis Assistant")
    
    # Your existing tabs
    tab1, tab2 = st.tabs(["Chat", "File Upload"])
    
    with tab1:
        # Your existing chat interface
        chat_interface()
    
    with tab2:
        # Your existing file upload interface
        file_upload_interface()

@app.post("/analyze")
async def analyze_data(request: AnalysisRequest):
    try:
        result = sql_analysis_system.analyze(request.query)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema")
async def get_schema():
    try:
        schema = code_search_tools.get_database_schema()
        return {"schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sql_analyze/", response_model=SQLAnalysisResponse)
async def analyze_sql_query(request: SQLAnalysisRequest):
    """Endpoint for SQL analysis"""
    try:
        initial_state = AnalysisState(
            messages=[HumanMessage(content=request.query)],
            code_search=None,
            doc_search=None,
            analysis=None,
            query=None,
            results=None,
            visualization=None,
            error=None,
            metadata={"start_time": time.time()}
        )
        
        result = sql_analysis_system.workflow.invoke(initial_state)
        execution_time = time.time() - initial_state["metadata"]["start_time"]
        
        if result.get("error"):
            raise HTTPException(
                status_code=400,
                detail=result["error"]
            )
            
        return SQLAnalysisResponse(
            result=result,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"SQL analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/database_schema/")
async def get_database_schema():
    """Get the database schema"""
    try:
        schema = sql_analysis_system.sql_db.get_schema()
        return JSONResponse({
            "status": "success",
            "schema": schema
        })
    except Exception as e:
        logger.error(f"Failed to get schema: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    main() 