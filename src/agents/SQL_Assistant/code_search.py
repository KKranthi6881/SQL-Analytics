from typing import Dict, List, Any
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from src.tools import SearchTools
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Simple search result structure"""
    content: List[str]
    relevant_tables: List[str]
    confidence: float
    search_results: Dict[str, Any]  # Store raw search results

class SimpleCodeSearch:
    """Simplified code search system"""
    
    def __init__(self, tools: SearchTools):
        self.search_tools = tools
        self.setup_components()

    def setup_components(self):
        """Initialize required components"""
        self.llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0,
            base_url="http://localhost:11434"
        )
        
         # Updated prompt template focused on SQL query relevant information
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert focusing on identifying relevant database structures and code patterns. 
            Your task is to analyze code examples and extract information that would be useful for SQL query generation.

            Focus ONLY on elements directly related to the user's question. Do not include unrelated tables or code.

            For each relevant piece of code:
            1. Identify tables and their columns that are specifically needed to answer the question
            2. Extract join conditions and relationships between these tables
            3. Find relevant SQL patterns or code examples that show how to:
            - Access the required data
            - Apply necessary filters
            - Calculate relevant metrics
            - Handle any special cases

            Ignore any tables or code patterns that aren't directly related to answering the user's question."""),
                        
                        ("human", """User Question: {question}

            Available Code Context:
            {code_context}

            Please analyze the code and provide:
            1. Only the tables and columns needed for this specific question
            2. Use only available columns from {code_context}
            3. Don't generalize or halluciate the columns or tables.           
            2. Only the relationships between these specific tables
            3. Relevant code examples showing how to work with this data
            4. Any important technical considerations for query generation

            Remember to focus ONLY on elements needed for this specific question.""")
                    ])

    def search_code(self, question: str) -> Dict[str, Any]:
        """Perform code search analysis"""
        try:
            # Use SearchTools to find relevant code
            search_results = self.search_tools.search_code(question)
            
            # Format code context
            code_snippets = []
            for result in search_results.get('results', []):
                code_snippets.append(
                    f"Source: {result.get('source', 'Unknown')}\n"
                    f"Code:\n{result.get('content', '')}\n"
                )
            
            code_context = "\n".join(code_snippets)
            
            # Get LLM analysis
            response = self.llm.invoke(
                self.prompt.format(
                    question=question,
                    code_context=code_context
                )
            )
            
            # Extract tables from search results
            tables = set()
            for result in search_results.get('results', []):
                content = result.get('content', '').upper()
                words = content.split()
                for i, word in enumerate(words):
                    if word in ['FROM', 'JOIN'] and i + 1 < len(words):
                        table = words[i + 1].strip('";,()').lower()
                        tables.add(table)
            
            # Create result
            result = SearchResult(
                content=[response.content],
                relevant_tables=list(tables),
                confidence=0.8,
                search_results=search_results
            )
            
            return {
                "status": "success",
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Code search failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add the project root directory to Python path
    project_root = str(Path(__file__).parent.parent.parent)
    sys.path.append(project_root)
    
    # Now import the required modules
    from src.tools import SearchTools
    from src.utils import ChromaDBManager
    
    # Initialize tools
    chroma_manager = ChromaDBManager()
    search_tools = SearchTools(chroma_manager)
    
    # Create searcher
    searcher = SimpleCodeSearch(search_tools)
    question = "Give me the list of top 10 movies by highest gross collection"
    
    result = searcher.search_code(question)
    if result["status"] == "success":
        print("\nAnalysis result:", result["result"].content[0])
        print("\nRelevant tables:", result["result"].relevant_tables)
        print("\nSearch confidence:", result["result"].confidence)
    else:
        print("Error:", result["error"]) 