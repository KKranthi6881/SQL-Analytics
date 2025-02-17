import streamlit as st
import requests
import json
from pathlib import Path
import os

# Set page config
st.set_page_config(
    page_title="Code Analysis System",
    page_icon="🔍",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8000"  # Adjust if your FastAPI server runs on a different port

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