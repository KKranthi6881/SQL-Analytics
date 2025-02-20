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
class DocSearchResult:
    """Simple document search result structure"""
    content: List[str]
    key_concepts: List[str]
    business_rules: List[str]
    confidence: float
    search_results: Dict[str, Any]

class SimpleDocSearch:
    """Simplified documentation search system"""
    
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
        
        # Define a simple prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a business analyst expert. Analyze the given question and documentation to identify:
1. Key business concepts and terminology
2. Business rules and requirements
3. Relevant workflows and processes

Keep your response focused and concise. Structure your response with clear sections."""),
            ("human", """Question: {question}

Documentation Context:
{doc_context},

Please analyze the doc and provide:
            1. Only the business context,related abbreviations,metadata info for the user specific question
            2. Use only available info from {doc_context}
            3. Don't generalize or halluciate the context, metadata and abbreviations .  """  )


        ])

    def search_documentation(self, question: str) -> Dict[str, Any]:
        """Perform documentation search and analysis"""
        try:
            # Use SearchTools to find relevant documentation
            search_results = self.search_tools.search_documentation(question)
            
            # Format documentation context
            doc_snippets = []
            for result in search_results.get('results', []):
                doc_snippets.append(
                    f"Source: {result.get('source', 'Unknown')}\n"
                    f"Content:\n{result.get('content', '')}\n"
                    f"Metadata: {result.get('metadata', {})}\n"
                )
            
            doc_context = "\n".join(doc_snippets)
            
            # Get LLM analysis
            response = self.llm.invoke(
                self.prompt.format(
                    question=question,
                    doc_context=doc_context
                )
            )
            
            # Extract key concepts and business rules
            response_text = response.content
            sections = response_text.split('\n')
            
            key_concepts = []
            business_rules = []
            
            current_section = None
            for line in sections:
                line = line.strip()
                if 'Key Concepts:' in line:
                    current_section = 'concepts'
                elif 'Business Rules:' in line:
                    current_section = 'rules'
                elif line and current_section:
                    if current_section == 'concepts':
                        key_concepts.append(line)
                    elif current_section == 'rules':
                        business_rules.append(line)
            
            # Create result
            result = DocSearchResult(
                content=[response_text],
                key_concepts=key_concepts,
                business_rules=business_rules,
                confidence=0.8,
                search_results=search_results
            )
            
            return {
                "status": "success",
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Documentation search failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def format_response(self, result: DocSearchResult) -> str:
        """Format the search result into a readable response"""
        return f"""
Documentation Analysis Results:

1. Key Concepts:
{chr(10).join(f'   • {concept}' for concept in result.key_concepts)}

2. Business Rules:
{chr(10).join(f'   • {rule}' for rule in result.business_rules)}

3. Detailed Analysis:
{result.content[0]}
"""

# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add the project root directory to Python path
    project_root = str(Path(__file__).parent.parent.parent)
    sys.path.append(project_root)
    
    from src.tools import SearchTools
    from src.utils import ChromaDBManager
    
    # Initialize tools
    chroma_manager = ChromaDBManager()
    search_tools = SearchTools(chroma_manager)
    
    # Create searcher
    searcher = SimpleDocSearch(search_tools)
    question = "Give me the list of top 10 movies by highest gross collection"
    
    result = searcher.search_documentation(question)
    if result["status"] == "success":
        formatted_response = searcher.format_response(result["result"])
        print("\nAnalysis Results:")
        print(formatted_response)
        print("\nConfidence:", result["result"].confidence)
    else:
        print("Error:", result["error"]) 