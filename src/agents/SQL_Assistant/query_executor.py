from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd
from sqlalchemy import create_engine, text
from src.agents.SQL_Assistant.query_evaluator import SQLQueryEvaluator, QueryEvaluation
from src.agents.SQL_Assistant.sql_generator import GeneratedSQL
import logging
import sqlglot
from sqlglot import expressions as exp
import sqlite3
import time
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOllama


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Structure to hold query execution results"""
    success: bool
    data: Optional[pd.DataFrame]
    error_message: Optional[str]
    execution_time: float
    row_count: int
    needs_revision: bool
    revision_reason: Optional[str]
    is_corrected: bool = False
    correction_details: Optional[Dict] = None
    query_used: Optional[str] = None

class SQLQueryExecutor:
    """Executes SQL queries and handles failures with evaluation feedback"""
    
    def __init__(self, db_connection_string: str, query_evaluator: SQLQueryEvaluator):
        """Initialize with database connection and evaluator"""
        self.engine = create_engine(db_connection_string)
        self.evaluator = query_evaluator
        self.db_schema = query_evaluator.db_schema  # Get schema from evaluator
        self.setup_components()  # Initialize LLM and prompts
        
    def setup_components(self):
        """Initialize components and prompts"""
        self.llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0,
            base_url="http://localhost:11434"
        )
        
        self.correction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL query correction system. Your task is to:
            1. Analyze SQL queries for syntax and logical errors
            2. Understand the database schema and relationships
            3. Fix any compatibility issues
            4. Ensure the query answers the original question
            
            Database Schema:
            {schema}
            
            Focus on:
            - SQLite syntax compatibility
            - Proper table relationships
            - Correct column references
            - Date/time function handling
            - Aggregation and grouping logic
            
            When providing the corrected query:
            1. Include the complete SQL statement
            2. Ensure proper formatting
            3. Include all necessary clauses
            4. End with a semicolon
            5. Do not include markdown code blocks
            """),
            
            ("human", """Original Question: {question}
            
            Original Query:
            {query}
            
            Evaluation Feedback:
            - Errors: {errors}
            - Warnings: {warnings}
            - Suggestions: {suggestions}
            
            Please analyze and respond with:
            NEEDS_CORRECTION: true/false
            ISSUES_FOUND: [list of issues]
            CORRECTED_QUERY:
            <write the complete corrected SQL query here>
            EXPLANATION: [why changes were made]
            CONFIDENCE: [0-1 score]
            """)
        ])

    def execute_query(self, generated_sql: GeneratedSQL) -> Dict[str, Any]:
        """Execute query with intelligent correction"""
        try:
            # Get initial evaluation
            evaluation = self.evaluator.evaluate_query(
                query=generated_sql.query,
                question=generated_sql.original_question,
                generated_sql=generated_sql
            )
            
            if evaluation["status"] == "error":
                return self._handle_error("Evaluation failed", evaluation["error"])
            
            eval_result = evaluation["result"]
            
            # Get correction analysis from LLM
            correction_response = self.llm.invoke(
                self.correction_prompt.format(
                    schema=self.db_schema,
                    question=generated_sql.original_question,
                    query=generated_sql.query,
                    errors=eval_result.errors,
                    warnings=eval_result.warnings,
                    suggestions=eval_result.suggestions
                )
            )
            
            # Parse correction response
            correction_result = self._parse_correction_response(correction_response.content)
            
            if correction_result["needs_correction"]:
                logger.info("Query needs correction:")
                logger.info(correction_result["explanation"])
                
                # Try corrected query
                corrected_query = correction_result["corrected_query"]
                return self._try_execute(
                    corrected_query,
                    is_correction=True,
                    correction_details=correction_result
                )
            else:
                # Try original query
                return self._try_execute(generated_sql.query)
                
        except Exception as e:
            return self._handle_error("Execution failed", str(e))

    def _try_execute(
        self,
        query: str,
        is_correction: bool = False,
        correction_details: Dict = None
    ) -> Dict[str, Any]:
        """Attempt to execute a query"""
        try:
            clean_query = self._clean_query(query)
            logger.info(f"Executing query:\n{clean_query}")
            
            # Create a new connection for each execution
            engine = create_engine(str(self.engine.url), echo=True)
            
            start_time = time.time()
            try:
                # Use pandas read_sql_query with the engine directly
                result = pd.read_sql_query(
                    sql=text(clean_query),  # Use SQLAlchemy text() for safe query execution
                    con=engine
                )
                execution_time = time.time() - start_time
                
                # Close the engine
                engine.dispose()
                
                return {
                    "status": "success",
                    "result": QueryResult(
                        success=True,
                        data=result,
                        error_message=None,
                        execution_time=execution_time,
                        row_count=len(result),
                        needs_revision=False,
                        revision_reason=None,
                        is_corrected=is_correction,
                        correction_details=correction_details,
                        query_used=clean_query
                    )
                }
                
            except Exception as e:
                # Make sure to dispose of the engine on error
                engine.dispose()
                raise e
                
        except Exception as e:
            logger.error(f"Execution attempt failed: {str(e)}")
            return self._handle_error("Execution failed", str(e))

    def _parse_correction_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM correction response"""
        result = {
            "needs_correction": False,
            "issues": [],
            "corrected_query": None,
            "explanation": "",
            "confidence": 0.0
        }
        
        current_section = None
        current_content = []
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('NEEDS_CORRECTION:'):
                result["needs_correction"] = line.split(':', 1)[1].strip().lower() == 'true'
            elif line.startswith('ISSUES_FOUND:'):
                result["issues"] = self._parse_list(line.split(':', 1)[1])
            elif line.startswith('CORRECTED_QUERY:'):
                current_section = 'query'
                current_content = []
            elif line.startswith('EXPLANATION:'):
                if current_section == 'query':
                    # Join the collected query lines
                    result["corrected_query"] = '\n'.join(current_content).strip()
                current_section = 'explanation'
                current_content = []
            elif line.startswith('CONFIDENCE:'):
                if current_section == 'explanation':
                    result["explanation"] = '\n'.join(current_content).strip()
                try:
                    result["confidence"] = float(line.split(':', 1)[1].strip())
                except:
                    result["confidence"] = 0.0
            elif current_section:
                # Collect content for current section
                if not line.startswith('```'):  # Skip code block markers
                    current_content.append(line)
        
        # Handle last section if it was explanation
        if current_section == 'explanation':
            result["explanation"] = '\n'.join(current_content).strip()
        
        # Clean up the corrected query if present
        if result["corrected_query"]:
            result["corrected_query"] = self._clean_query(result["corrected_query"])
        
        logger.info(f"Parsed correction result:")
        logger.info(f"Needs correction: {result['needs_correction']}")
        logger.info(f"Corrected query: {result['corrected_query']}")
        
        return result

    def _parse_list(self, text: str) -> List[str]:
        """Parse a string into a list of items"""
        # Remove brackets and split by commas
        text = text.strip('[]')
        items = [item.strip().strip('"\'') for item in text.split(',')]
        return [item for item in items if item]

    def _analyze_results(
        self,
        df: pd.DataFrame,
        execution_time: float,
        generated_sql: GeneratedSQL,
        evaluation: QueryEvaluation
    ) -> QueryResult:
        """Analyze query results and determine if revision is needed"""
        needs_revision = False
        revision_reason = None
        
        # Check if results are empty
        if df.empty:
            needs_revision = True
            revision_reason = "Query returned no results"
        
        # Check if we got too many results
        elif len(df) > 10000:  # Arbitrary limit
            needs_revision = True
            revision_reason = "Query returned too many results, might need additional filters"
        
        # Check if execution time is too long
        elif execution_time > 30:  # 30 seconds threshold
            needs_revision = True
            revision_reason = "Query execution took too long, might need optimization"
        
        # Check if we got the expected columns
        expected_columns = self._extract_expected_columns(generated_sql)
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            needs_revision = True
            revision_reason = f"Missing expected columns: {missing_columns}"
        
        return QueryResult(
            success=True,
            data=df,
            error_message=None,
            execution_time=execution_time,
            row_count=len(df),
            needs_revision=needs_revision,
            revision_reason=revision_reason
        )

    def _analyze_error(self, error_message: str) -> str:
        """Analyze error message and provide meaningful revision reason"""
        error_patterns = {
            "column": "Column reference error",
            "relation": "Table reference error",
            "syntax": "Syntax error",
            "permission": "Permission error",
            "duplicate": "Duplicate column reference",
            "ambiguous": "Ambiguous column reference",
            "type": "Data type mismatch",
            "constraint": "Constraint violation"
        }
        
        error_message = error_message.lower()
        for pattern, reason in error_patterns.items():
            if pattern in error_message:
                return f"{reason}: {error_message}"
        
        return error_message

    def _extract_expected_columns(self, generated_sql: GeneratedSQL) -> List[str]:
        """Extract expected column names from the query"""
        try:
            # Parse the query to get selected columns
            parsed = sqlglot.parse_one(generated_sql.query)
            columns = []
            
            # Get column names from SELECT clause
            for select in parsed.find_all(exp.Select):
                for col in select.expressions:
                    if hasattr(col, 'alias'):
                        columns.append(col.alias)
                    elif hasattr(col, 'name'):
                        columns.append(col.name)
                    
            return columns
        except:
            # If parsing fails, return empty list
            return []

    def revise_and_retry(self, generated_sql: GeneratedSQL, revision_reason: str) -> Dict[str, Any]:
        """Attempt to revise and retry a failed query"""
        try:
            # Get correction from LLM
            correction_response = self.llm.invoke(
                self.correction_prompt.format(
                    schema=self.db_schema,
                    question=generated_sql.original_question,
                    query=generated_sql.query,
                    errors=[revision_reason],
                    warnings=[],
                    suggestions=[]
                )
            )
            
            correction_result = self._parse_correction_response(correction_response.content)
            
            if correction_result["needs_correction"] and correction_result["corrected_query"]:
                return self._try_execute(
                    correction_result["corrected_query"],
                    is_correction=True,
                    correction_details=correction_result
                )
            else:
                return self._handle_error(
                    "Revision failed",
                    "Could not generate valid correction"
                )
            
        except Exception as e:
            return self._handle_error("Revision failed", str(e))

    def _modify_query(
        self,
        query: str,
        revision_reason: str,
        evaluation: QueryEvaluation
    ) -> Optional[str]:
        """Attempt to modify the query based on revision reason"""
        try:
            parsed = sqlglot.parse_one(query)
            
            if "too many results" in revision_reason.lower():
                # Add LIMIT if missing
                if not parsed.find(exp.Limit):
                    return str(parsed) + " LIMIT 1000"
                    
            elif "took too long" in revision_reason.lower():
                # Add indexes or optimize joins
                pass  # Would need more complex logic
                
            elif "no results" in revision_reason.lower():
                # Try removing some WHERE conditions
                where = parsed.find(exp.Where)
                if where:
                    # Remove the last condition
                    conditions = where.find_all(exp.And)
                    if conditions:
                        conditions[-1].pop()
                        return str(parsed)
            
            return None
            
        except Exception as e:
            logger.error(f"Query modification failed: {str(e)}")
            return None

    def _handle_error(self, error_type: str, error_details: str) -> Dict[str, Any]:
        """Standardized error handling"""
        logger.error(f"{error_type}: {error_details}")
        return {
            "status": "error",
            "error": f"{error_type}",
            "needs_revision": True,
            "revision_reason": error_details
        }

    def _clean_query(self, query: str) -> str:
        """Clean the query for execution"""
        # Remove code block markers
        clean_query = query.replace('```sql', '').replace('```', '').strip()
        
        # Remove any leading/trailing whitespace from lines
        clean_query = '\n'.join(line.strip() for line in clean_query.split('\n'))
        
        # Ensure query ends with semicolon
        if not clean_query.rstrip().endswith(';'):
            clean_query += ';'
        
        return clean_query

# Example usage
if __name__ == "__main__":
    from src.agents.SQL_Assistant.sql_generator import SQLQueryGenerator
    from src.agents.SQL_Assistant.question_parser import QuestionParser
    from src.agents.SQL_Assistant.doc_search import SimpleDocSearch
    from src.agents.SQL_Assistant.code_search import SimpleCodeSearch
    from src.tools import SearchTools
    from src.utils import ChromaDBManager
    
    # Get schema from SQLite database
    sqlite_path = "/Users/Kranthi_1/SQL-Analytics/src/db/sampledb/sakila_master.db"
    
    def get_db_schema():
        schema = {}
        with sqlite3.connect(sqlite_path) as conn:
            # Get all tables
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            
            for table in tables:
                table_name = table[0]
                # Get columns for each table
                columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                
                # Get foreign keys
                foreign_keys = conn.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()
                
                relationships = []
                for fk in foreign_keys:
                    relationships.append({
                        'table': fk[2],  # referenced table
                        'keys': [fk[3], fk[4]]  # [from_col, to_col]
                    })
                
                schema[table_name] = {
                    'columns': [col[1] for col in columns],  # column names
                    'relationships': relationships
                }
        
        return schema
    
    # Initialize components
    db_schema = get_db_schema()
    
    # Setup components
    chroma_manager = ChromaDBManager()
    search_tools = SearchTools(chroma_manager)
    
    parser = QuestionParser(
        SimpleDocSearch(search_tools),
        SimpleCodeSearch(search_tools)
    )
    sql_generator = SQLQueryGenerator()
    evaluator = SQLQueryEvaluator(db_schema)
    
    # Create executor with SQLite connection string
    executor = SQLQueryExecutor(
        f"sqlite:///{sqlite_path}",
        evaluator
    )
    
    # Test questions
    test_questions = [
        "How does the rental rate vary by film rating (G, PG, PG-13, R, NC-17)?"
    ]
    
    for question in test_questions:
        print(f"\nProcessing question: {question}")
        print("-" * 50)
        
        # Process pipeline
        parsed = parser.parse_question(question)
        if parsed["status"] == "success":
            generated = sql_generator.generate_query(parsed["result"], question)
            if generated["status"] == "success":
                result = executor.execute_query(generated["result"])
                
                if result["status"] == "success":
                    query_result = result["result"]
                    if query_result.success:
                        print("\nQuery Results:")
                        print(query_result.data)
                        print(f"\nExecution Time: {query_result.execution_time:.2f} seconds")
                        print(f"Row Count: {query_result.row_count}")
                        
                        if query_result.needs_revision:
                            print(f"\nQuery needs revision: {query_result.revision_reason}")
                            # Try revision
                            revised = executor.revise_and_retry(
                                generated["result"],
                                query_result.revision_reason
                            )
                            if revised["status"] == "success":
                                print("\nRevised Query Results:")
                                print(revised["result"].data)
                    else:
                        print(f"Query failed: {query_result.error_message}")
                else:
                    print(f"Execution error: {result['error']}")
                    if result.get('needs_revision'):
                        print(f"Revision needed: {result['revision_reason']}")
            else:
                print(f"Generation error: {generated['error']}")
        else:
            print(f"Parsing error: {parsed['error']}") 