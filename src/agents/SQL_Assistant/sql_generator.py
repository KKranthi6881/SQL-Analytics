from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from src.agents.SQL_Assistant.question_parser import ParsedQuestion
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sql_generator.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class GeneratedSQL:
    """Structure to hold the generated SQL query and its metadata"""
    query: str
    explanation: str
    tables_used: List[str]
    parameters: Dict[str, Any]
    confidence: float
    warnings: List[str]
    original_question: str

class SQLQueryGenerator:
    """Generates SQL queries based on parsed question analysis"""
    
    def __init__(self):
        self.setup_components()

    def setup_components(self):
        """Initialize required components"""
        self.llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0,
            base_url="http://localhost:11434"
        )
        
        self.sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL query generator with SQLite database. Generate precise SQL queries with SQLite syntax by using ONLY the tables and columns available in the provided schema.

            Important Rules:
            1. ONLY use tables and columns that exist in the technical context
            2. Do NOT assume or hallucinate columns that aren't mentioned
            3. Use proper JOINs when counting or aggregating across tables
            4. Always verify column existence before using in queries
            5. Follow the relationships defined in the technical context

            Respond in this format:
            QUERY:
            <your SQL query here>

            EXPLANATION:
            <explain how the query works>

            PARAMETERS:
            <any parameters needed>

            WARNINGS:
            <any potential issues>
            """),
            
            ("human", """Business Context:
            {business_context}
            
            Technical Context (Available Tables and Columns):
            {technical_context}
            
            User Question:
            {question}
            
            Generate a SQL query that:
            1. Only uses existing tables and columns
            2. Follows the defined relationships
            3. Correctly answers the user's question
            4. Must FOllow SQLite syntax only. Don't follow any other SQL syntax.""")
        ])

    def generate_query(self, parsed_question: ParsedQuestion, question: str) -> Dict[str, Any]:
        """Generate SQL query based on parsed question analysis"""
        try:
            # Log input data
            logger.info(f"Generating SQL for question: {question}")
            logger.info("Business Context:")
            logger.info(f"Key Concepts: {parsed_question.business_context.get('key_concepts', [])}")
            logger.info(f"Business Rules: {parsed_question.business_context.get('business_rules', [])}")
            
            logger.info("Technical Context:")
            logger.info(f"Relevant Tables: {parsed_question.technical_context.get('relevant_tables', [])}")
            logger.info(f"Table Relationships: {parsed_question.technical_context.get('table_relationships', [])}")
            
            # Extract contexts from parsed question
            business_context = {
                "concepts": parsed_question.business_context["key_concepts"],
                "rules": parsed_question.business_context["business_rules"],
                "analysis": parsed_question.business_context["detailed_analysis"]
            }
            
            technical_context = {
                "tables": parsed_question.technical_context["relevant_tables"],
                "relationships": parsed_question.technical_context["table_relationships"],
                "analysis": parsed_question.technical_context["technical_analysis"]
            }
            
            # Log formatted contexts
            logger.info("Formatted Context for LLM:")
            logger.info(f"Business Context: {business_context}")
            logger.info(f"Technical Context: {technical_context}")
            
            # Get SQL generation response
            response = self.llm.invoke(
                self.sql_prompt.format(
                    business_context=str(business_context),
                    technical_context=str(technical_context),
                    question=question
                )
            )
            
            # Log LLM response
            logger.info("LLM Response:")
            logger.info(response.content)
            
            # Parse the response
            parsed_response = self._parse_sql_response(response.content)
            logger.info("Parsed Response:")
            logger.info(parsed_response)
            
            # Extract tables and log
            tables_used = self._extract_tables_from_query(parsed_response["query"])
            logger.info(f"Extracted Tables: {tables_used}")
            
            # Create result
            result = GeneratedSQL(
                query=parsed_response["query"],
                explanation=parsed_response["explanation"],
                tables_used=tables_used,
                parameters=parsed_response["parameters"],
                confidence=parsed_question.confidence * 0.9,
                warnings=parsed_response["warnings"],
                original_question=question
            )
            
            logger.info("Generated SQL Result:")
            logger.info(f"Query: {result.query}")
            logger.info(f"Tables Used: {result.tables_used}")
            logger.info(f"Confidence: {result.confidence}")
            
            return {
                "status": "success",
                "result": result
            }
            
        except Exception as e:
            logger.error(f"SQL generation failed: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

    def _parse_sql_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        logger.info("Raw LLM Response:")
        logger.info(response)
        
        # Initialize default structure
        result = {
            "query": "",
            "explanation": "",
            "parameters": {},
            "warnings": []
        }
        
        try:
            # If response is just a SQL query (no formatting)
            if response.strip().upper().startswith('SELECT'):
                result["query"] = response.strip()
                result["explanation"] = "Direct SQL query generated"
                return result
            
            # Otherwise try to parse formatted response
            current_section = None
            current_content = []
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.upper().startswith('QUERY:'):
                    current_section = 'query'
                    continue
                elif line.upper().startswith('EXPLANATION:'):
                    if current_section == 'query':
                        result['query'] = '\n'.join(current_content).strip()
                    current_section = 'explanation'
                    current_content = []
                    continue
                elif line.upper().startswith('PARAMETERS:'):
                    if current_section == 'explanation':
                        result['explanation'] = '\n'.join(current_content).strip()
                    current_section = 'parameters'
                    current_content = []
                    continue
                elif line.upper().startswith('WARNINGS:'):
                    if current_section == 'parameters':
                        # Parse parameters if any
                        for param in current_content:
                            if ':' in param:
                                key, value = param.split(':', 1)
                                result['parameters'][key.strip()] = value.strip()
                    current_section = 'warnings'
                    current_content = []
                    continue
                
                if current_section:
                    current_content.append(line)
            
            # Handle last section
            if current_section == 'warnings':
                result['warnings'] = [w.strip('- ') for w in current_content if w.strip()]
            
            # If we got no query but have content, assume it's a SQL query
            if not result["query"] and current_content:
                result["query"] = '\n'.join(current_content).strip()
            
            logger.info("Parsed Response:")
            logger.info(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}", exc_info=True)
            # Return the raw response as query if parsing fails
            return {
                "query": response.strip(),
                "explanation": "Direct response from LLM",
                "parameters": {},
                "warnings": ["Response parsing failed"]
            }

    def _extract_tables_from_query(self, query: str) -> List[str]:
        """Extract table names from SQL query"""
        tables = set()
        words = query.upper().split()
        
        for i, word in enumerate(words):
            if word in ['FROM', 'JOIN'] and i + 1 < len(words):
                table = words[i + 1].strip('";,()').lower()
                tables.add(table)
        
        return list(tables)

    def validate_query(self, query: str) -> List[str]:
        """Basic SQL query validation"""
        warnings = []
        
        # Check for basic SQL injection risks
        risky_patterns = ["--", ";", "/*", "*/", "UNION", "DROP", "DELETE", "UPDATE"]
        for pattern in risky_patterns:
            if pattern in query.upper():
                warnings.append(f"Potential SQL injection risk: {pattern}")
        
        # Check for SELECT *
        if "SELECT *" in query.upper():
            warnings.append("Using SELECT * is not recommended for production queries")
        
        # Check for basic structure
        if "FROM" not in query.upper():
            warnings.append("Query missing FROM clause")
        
        return warnings

# Example usage
if __name__ == "__main__":
    from src.agents.SQL_Assistant.question_parser import QuestionParser
    from src.agents.SQL_Assistant.doc_search import SimpleDocSearch
    from src.agents.SQL_Assistant.code_search import SimpleCodeSearch
    from src.tools import SearchTools
    from src.utils import ChromaDBManager
    
    # Initialize components
    chroma_manager = ChromaDBManager()
    search_tools = SearchTools(chroma_manager)
    
    # Create parser
    parser = QuestionParser(
        SimpleDocSearch(search_tools),
        SimpleCodeSearch(search_tools)
    )
    
    # Create SQL generator
    sql_generator = SQLQueryGenerator()
    
    # Test question
    question = "Give me the list of top 10 movies by highest gross collection"
    
    # Parse question
    parsed = parser.parse_question(question)
    if parsed["status"] == "success":
        # Generate SQL
        result = sql_generator.generate_query(parsed["result"], question)
        
        if result["status"] == "success":
            generated = result["result"]
            print("\nGenerated SQL Query:")
            print(generated.query)
            print("\nExplanation:")
            print(generated.explanation)
            print("\nTables Used:", generated.tables_used)
            print("\nParameters:", generated.parameters)
            print("\nWarnings:", generated.warnings)
            print("\nConfidence Score:", generated.confidence)
        else:
            print("Error:", result["error"])
    else:
        print("Error:", parsed["error"]) 