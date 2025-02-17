from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
from .utils import ChromaDBManager
from .tools import SearchTools
from .agents.code_research import SimpleAnalysisSystem
import logging
from pydantic import BaseModel

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
tools = SearchTools(db_manager)
analysis_system = SimpleAnalysisSystem(tools)

# Add new model for code analysis requests
class CodeAnalysisRequest(BaseModel):
    query: str

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
            docs = db_manager.process_pdf(str(file_path))
        elif file_extension == '.sql':
            logger.info("Processing SQL file...")
            docs = db_manager.process_sql(str(file_path))
        elif file_extension == '.py':
            logger.info("Processing Python file...")
            docs = db_manager.process_python(str(file_path))
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