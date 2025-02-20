from dataclasses import dataclass
from typing import Dict, List, Any
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from src.agents.SQL_Assistant.doc_search import SimpleDocSearch, DocSearchResult
from src.agents.SQL_Assistant.code_search import SimpleCodeSearch, SearchResult
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ParsedQuestion:
    """Structure to hold the parsed question analysis"""
    business_context: Dict[str, Any]  # Business rules, concepts, requirements
    technical_context: Dict[str, Any]  # Tables, relationships, SQL patterns
    confidence: float
    raw_results: Dict[str, Any]  # Store raw search results

class QuestionParser:
    """
    Analyzes and combines documentation and code search results
    to provide comprehensive context for SQL generation
    """
    
    def __init__(self, doc_searcher: SimpleDocSearch, code_searcher: SimpleCodeSearch):
        self.doc_searcher = doc_searcher
        self.code_searcher = code_searcher
        self.setup_components()

    def setup_components(self):
        """Initialize required components"""
        self.llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0,
            base_url="http://localhost:11434"
        )
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL analyst who combines business requirements with technical implementation.
            Analyze both business documentation and technical code to create a comprehensive context for SQL query generation.
            
            Focus on:
            1. Mapping business requirements to technical implementation
            2. Identifying required tables and their business purpose
            3. Understanding business rules that affect query logic
            4. Noting any special cases or conditions
            
            Provide a structured analysis that clearly connects business needs to technical implementation."""),
            
            ("human", """Business Context:
            {business_context}
            
            Technical Context:
            {technical_context}
            
            User Question:
            {question}
            
            Provide a comprehensive analysis that includes:
            1. Business Requirements
            2. Technical Implementation Details
            3. Required Tables and Their Purpose
            4. Important Business Rules for Query Logic""")
        ])

    def parse_question(self, question: str) -> Dict[str, Any]:
        """
        Analyze the question using both documentation and code search,
        then combine the results into a comprehensive context
        """
        try:
            # Perform both searches
            doc_result = self.doc_searcher.search_documentation(question)
            code_result = self.code_searcher.search_code(question)
            
            if doc_result["status"] == "error" or code_result["status"] == "error":
                raise Exception("Search failed: " + 
                              doc_result.get("error", "") + " " + 
                              code_result.get("error", ""))
            
            # Format the contexts for analysis
            doc_data: DocSearchResult = doc_result["result"]
            code_data: SearchResult = code_result["result"]
            
            business_context = self.doc_searcher.format_response(doc_data)
            technical_context = (f"Relevant Tables: {', '.join(code_data.relevant_tables)}\n"
                               f"Technical Analysis: {code_data.content[0]}")
            
            # Get combined analysis
            analysis = self.llm.invoke(
                self.analysis_prompt.format(
                    business_context=business_context,
                    technical_context=technical_context,
                    question=question
                )
            )
            
            # Calculate combined confidence
            combined_confidence = (doc_data.confidence + code_data.confidence) / 2
            
            # Create structured result
            parsed_result = ParsedQuestion(
                business_context={
                    "key_concepts": doc_data.key_concepts,
                    "business_rules": doc_data.business_rules,
                    "detailed_analysis": doc_data.content[0]
                },
                technical_context={
                    "relevant_tables": code_data.relevant_tables,
                    "technical_analysis": code_data.content[0],
                    "table_relationships": self._extract_relationships(code_data)
                },
                confidence=combined_confidence,
                raw_results={
                    "doc_search": doc_data,
                    "code_search": code_data,
                    "combined_analysis": analysis.content
                }
            )
            
            return {
                "status": "success",
                "result": parsed_result
            }
            
        except Exception as e:
            logger.error(f"Question parsing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _extract_relationships(self, code_data: SearchResult) -> List[Dict[str, str]]:
        """Extract table relationships from code search results"""
        relationships = []
        content = code_data.content[0].lower()
        
        # Simple relationship extraction from JOIN conditions
        for line in content.split('\n'):
            if 'join' in line and 'on' in line:
                parts = line.split('on')
                if len(parts) > 1:
                    relationships.append({
                        "condition": parts[1].strip(),
                        "join_type": "JOIN" if "join" in parts[0].upper() else "UNKNOWN"
                    })
        
        return relationships

# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add the project root directory to Python path
    project_root = str(Path(__file__).parent.parent.parent)
    sys.path.append(project_root)
    
    from src.tools import SearchTools
    from src.utils import ChromaDBManager
    
    # Initialize components
    chroma_manager = ChromaDBManager()
    search_tools = SearchTools(chroma_manager)
    
    # Create searchers
    doc_searcher = SimpleDocSearch(search_tools)
    code_searcher = SimpleCodeSearch(search_tools)
    
    # Create parser
    parser = QuestionParser(doc_searcher, code_searcher)
    
    # Test question
    question = "Give me the list of top 10 movies by highest gross collection"
    
    # Parse question
    result = parser.parse_question(question)
    if result["status"] == "success":
        parsed = result["result"]
        print("\nBusiness Context:")
        print("Key Concepts:", parsed.business_context["key_concepts"])
        print("Business Rules:", parsed.business_context["business_rules"])
        
        print("\nTechnical Context:")
        print("Relevant Tables:", parsed.technical_context["relevant_tables"])
        print("Table Relationships:", parsed.technical_context["table_relationships"])
        
        print("\nCombined Analysis:")
        print(parsed.raw_results["combined_analysis"])
        
        print("\nConfidence Score:", parsed.confidence)
    else:
        print("Error:", result["error"]) 