from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
from .utils import ChromaDBManager
import logging

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