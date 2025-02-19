from typing import Dict, List, Any, Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path
import uuid
import logging
import json
from sqlite3 import connect
from threading import Lock
import time
import sqlite3
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

from src.tools import SearchTools
from src.db.database import ChatDatabase
from src.utils import ChromaDBManager
from langchain_experimental.utilities import PythonREPL
from langchain.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from src.agents.code_research import SimpleAnalysisSystem

# Set up logger
logger = logging.getLogger(__name__)

# Constants for database paths
SAKILA_DB_PATH = "/Users/Kranthi_1/SQL-Analytics/src/db/sampledb/sakila_master.db"  # For SQL queries
CHAT_HISTORY_DB_PATH = str(Path(__file__).parent.parent.parent / "chat_history.db")  # For conversations
CHECKPOINT_DB_PATH = str(Path(__file__).parent.parent.parent / "chat_history.db")     # For langgraph checkpoints

# Define state type
class AnalystState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    code_context: Annotated[Dict, "Code search results"]
    doc_context: Annotated[Dict, "Documentation search results"]
    question_analysis: Annotated[Dict, "Parsed question and requirements"]
    sql_generation: Annotated[Dict, "Generated SQL query"]
    sql_validation: Annotated[Dict, "Validated SQL query"]
    query_results: Annotated[Dict, "SQL query execution results"]
    visualization_spec: Annotated[Dict, "Visualization specification"]
    final_output: Annotated[Dict, "Final formatted output with visualization"]

# Define structured outputs
class QuestionAnalysis(BaseModel):
    intent: str = Field(description="The main intent of the question")
    required_tables: List[str] = Field(description="Required tables for analysis")
    required_columns: List[str] = Field(description="Required columns for analysis")
    aggregations: List[str] = Field(description="Required aggregations")
    filters: List[str] = Field(description="Required filters")
    visualization_type: str = Field(description="Suggested visualization type")

class SQLGeneration(BaseModel):
    query: str = Field(description="Generated SQL query")
    explanation: str = Field(description="Explanation of the query")
    expected_output: str = Field(description="Expected output format")

class VisualizationSpec(BaseModel):
    type: str = Field(description="Type of visualization")
    x_axis: str = Field(description="X-axis specification")
    y_axis: str = Field(description="Y-axis specification")
    title: str = Field(description="Chart title")
    additional_params: Dict = Field(description="Additional visualization parameters")

class SQLValidation(BaseModel):
    is_valid: bool = Field(description="Whether the SQL is valid")
    issues: List[str] = Field(description="List of identified issues")
    fixed_query: str = Field(description="Fixed SQL query if issues were found")
    table_dependencies: List[str] = Field(description="Tables used in the query")
    column_dependencies: List[str] = Field(description="Columns used in the query")

class QueryResult(BaseModel):
    data: List[Dict] = Field(description="Query result data")
    column_names: List[str] = Field(description="Names of columns in result")
    row_count: int = Field(description="Number of rows returned")
    execution_time: float = Field(description="Query execution time in seconds")

class CodeAnalysisResult(BaseModel):
    tables: List[str] = Field(description="Tables identified in code")
    relationships: List[str] = Field(description="Table relationships found")
    business_logic: str = Field(description="Business logic explanation")
    technical_details: str = Field(description="Technical implementation details")

class DocAnalysisResult(BaseModel):
    key_concepts: List[str] = Field(description="Key business concepts")
    workflows: List[str] = Field(description="Business workflows")
    requirements: str = Field(description="Business requirements")
    context: str = Field(description="Additional business context")

# Add near the top with other models
class DataAnalysisRequest(BaseModel):
    query: str = Field(description="The analysis question or query")

# Add DatabaseConnection class
class DatabaseConnection:
    def __init__(self, db_path: str = SAKILA_DB_PATH):  # Use Sakila DB by default
        self.db_path = db_path
        self._validate_database()
        logger.info(f"Initialized database connection to: {self.db_path}")

    def _validate_database(self):
        """Validate database connection and schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all tables
                tables = pd.read_sql_query("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """, conn)
                
                if tables.empty:
                    raise ValueError(f"No tables found in database: {self.db_path}")
                
                self.tables = tables['name'].tolist()
                
                # Get schema for each table
                self.schema = {}
                for table in self.tables:
                    schema = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
                    self.schema[table] = schema.to_dict('records')
                
                logger.info(f"Successfully connected to database with {len(self.tables)} tables")
                
        except Exception as e:
            logger.error(f"Database validation failed: {str(e)}")
            raise

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    def get_schema(self) -> Dict[str, Any]:
        """Get database schema information."""
        return {
            "tables": self.tables,
            "schema": self.schema
        }

# Add DataAnalysisSystem class
class DataAnalysisSystem:
    def __init__(self, tools: Optional[SearchTools] = None, db_path: str = SAKILA_DB_PATH):
        # Database for SQL queries
        self.db_conn = DatabaseConnection(db_path)
        
        # Database for chat history
        self.chat_db = ChatDatabase(CHAT_HISTORY_DB_PATH)
        
        # Create the agent with proper checkpointing
        self.app = create_analyst_agent(tools, self.db_conn)
        
        self.python_repl = PythonREPL()
        self._lock = Lock()

    def analyze(self, query: str) -> Dict[str, Any]:
        """Process a query through the analysis workflow."""
        try:
            conversation_id = str(uuid.uuid4())
            logger.info(f"Starting new analysis {conversation_id}")
            logger.info(f"Query: {query}")
            
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "code_context": {},
                "doc_context": {},
                "question_analysis": {},
                "sql_generation": {},
                "sql_validation": {},
                "query_results": {},
                "visualization_spec": {},
                "final_output": {}
            }
            logger.info("Initial state created")
            
            config = {
                "configurable": {
                    "thread_id": conversation_id
                }
            }
            logger.info("Workflow configuration set")
            
            logger.info("Invoking analysis workflow")
            with self._lock:
                result = self.app.invoke(initial_state, config)
            
            logger.info("Analysis workflow completed")
            
            # Create a serializable version of the result
            serializable_result = {
                "messages": [msg.content for msg in result.get("messages", [])],
                "question_analysis": result.get("question_analysis", {}),
                "sql_generation": result.get("sql_generation", {}),
                "sql_validation": result.get("sql_validation", {}),
                "query_results": result.get("query_results", {}),
                "visualization_spec": result.get("visualization_spec", {}),
                "final_output": result.get("final_output", {})
            }
            logger.debug(f"Final result: {json.dumps(serializable_result, indent=2)}")
            
            if "error" in result:
                logger.error(f"Analysis failed: {result['error']}")
                return {
                    "error": result["error"],
                    "suggestion": result.get("suggestion", "Please try rephrasing your question"),
                    "details": result.get("details", "")
                }
            
            return serializable_result
            
        except Exception as e:
            logger.error(f"Unexpected error in analysis: {str(e)}", exc_info=True)
            return {
                "error": "Analysis system failed",
                "suggestion": "Please try again later",
                "details": str(e)
            }

    def _save_analysis(self, conversation_id: str, query: str, result: Dict):
        """Save analysis results to chat history database."""
        with self._lock:
            self.chat_db.save_conversation(
                conversation_id,
                {
                    "query": query,
                    "analysis": result.get("question_analysis", {}),
                    "sql_query": result.get("sql_validation", {}).get("query", ""),
                    "results": result.get("query_results", {}),
                    "visualization": result.get("visualization_spec", {}),
                    "summary": result.get("final_output", {}).get("summary", ""),
                    "timestamp": str(datetime.now())
                }
            )

def create_analyst_agent(tools: Optional[SearchTools], db_connection: DatabaseConnection):
    """Create the SQL analysis agent with proper graph configuration."""
    
    # Initialize LLM
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
        base_url="http://localhost:11434",
    )
    
    def analyze_question(state: AnalystState) -> Dict:
        nonlocal tools  # Make tools available in the function
        try:
            logger.info("Starting question analysis step")
            messages = state['messages']
            query = messages[-1].content
            logger.info(f"Processing question: {query}")
            
            # Get database schema
            schema = db_connection.get_schema()
            logger.info(f"Retrieved schema with {len(schema.get('tables', []))} tables")
            
            # Get code and doc context using search tools
            if tools:
                logger.info("Searching code and documentation context")
                code_results = tools.search_code(query)
                doc_results = tools.search_docs(query)
                logger.info(f"Found {len(code_results)} code snippets and {len(doc_results)} doc matches")
            else:
                code_results = []
                doc_results = []
                logger.info("No search tools available, skipping context search")
            
            # Define the prompt with context
            prompt = PromptTemplate.from_template("""
            You are a SQL expert. Analyze this question and return ONLY a JSON object.
            
            QUESTION: {question}
            
            DATABASE SCHEMA:
            Tables: {tables}
            Details: {schema}
            
            CODE CONTEXT:
            {code_context}
            
            DOCUMENTATION CONTEXT:
            {doc_context}
            
            For the question "give me the top 10 actors who has most movies", the exact response would be:
            {{"intent":"Find actors with highest number of movies","required_tables":["actor","film_actor"],"required_columns":["actor.first_name","actor.last_name","film_actor.film_id"],"aggregations":["COUNT(film_actor.film_id)"],"filters":["LIMIT 10","GROUP BY actor.actor_id","ORDER BY count DESC"],"visualization_type":"bar"}}

            Return a similar single-line JSON object for the current question.
            DO NOT include any other text, markdown, or explanations.
            DO NOT format the JSON with newlines or indentation.
            ONLY return the JSON object itself.
            """)
            
            # Get LLM response
            response = llm.invoke(
                prompt.format(
                    question=query,
                    tables=", ".join(schema.get("tables", [])),
                    schema=json.dumps(schema, indent=2),
                    code_context="\n".join(code_results) if code_results else "No relevant code found",
                    doc_context="\n".join(doc_results) if doc_results else "No relevant documentation found"
                )
            )
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            logger.info("Got LLM response for question analysis")
            logger.debug(f"Raw response: {response_text}")
            
            try:
                # Clean and parse the response
                cleaned_text = response_text.strip()
                
                # Remove any markdown or extra text
                if "{" in cleaned_text:
                    start_idx = cleaned_text.find("{")
                    end_idx = cleaned_text.rfind("}")
                    if end_idx == -1:  # No closing brace found
                        # Add closing brace if missing
                        cleaned_text = cleaned_text[start_idx:] + "}"
                    else:
                        cleaned_text = cleaned_text[start_idx:end_idx + 1]
                
                # Ensure valid JSON structure
                try:
                    # First try to parse as is
                    json_response = json.loads(cleaned_text)
                except json.JSONDecodeError:
                    # If that fails, try to fix common formatting issues
                    cleaned_text = cleaned_text.replace('\n', '')
                    cleaned_text = cleaned_text.replace('  ', '')
                    if not cleaned_text.endswith('}'):
                        cleaned_text += '}'
                    json_response = json.loads(cleaned_text)
                
                # Validate required fields
                required_fields = ["intent", "required_tables", "required_columns", 
                                 "aggregations", "filters", "visualization_type"]
                for field in required_fields:
                    if field not in json_response:
                        raise ValueError(f"Missing required field: {field}")
                
                # Create QuestionAnalysis object
                analysis = QuestionAnalysis(
                    intent=json_response["intent"],
                    required_tables=json_response["required_tables"],
                    required_columns=json_response["required_columns"],
                    aggregations=json_response["aggregations"],
                    filters=json_response["filters"],
                    visualization_type=json_response["visualization_type"]
                )
                
                logger.info(f"Successfully parsed question analysis: {analysis.dict()}")
                return {"question_analysis": analysis.dict()}
                
            except Exception as e:
                logger.error(f"Failed to parse analysis: {str(e)}")
                logger.error(f"Raw response was: {response_text}")
                return {
                    "error": "Failed to analyze question",
                    "suggestion": "Please try being more specific",
                    "details": str(e)
                }
                
        except Exception as e:
            logger.error(f"Question analysis failed: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "suggestion": "Please try rephrasing your question",
                "details": "Analysis system error"
            }

    # Create parsers
    question_parser = PydanticOutputParser(pydantic_object=QuestionAnalysis)
    sql_parser = PydanticOutputParser(pydantic_object=SQLGeneration)
    viz_parser = PydanticOutputParser(pydantic_object=VisualizationSpec)

    # Define the analysis workflow
    def generate_sql(state: AnalystState) -> Dict:
        """Generate SQL based on question analysis."""
        try:
            logger.info("Starting SQL generation step")
            analysis = state.get('question_analysis', {})
            logger.info(f"Generating SQL for analysis: {json.dumps(analysis, indent=2)}")
            
            if not analysis or 'error' in analysis:
                return {
                    "error": "Cannot generate SQL: Invalid question analysis",
                    "details": analysis.get('error', 'No valid analysis available')
                }

            schema = db_connection.get_schema()
            
            prompt = PromptTemplate.from_template("""
            Generate a SQL query for the following analysis:
            
            QUESTION ANALYSIS:
            Intent: {intent}
            Required Tables: {tables}
            Required Columns: {columns}
            Aggregations: {aggregations}
            Filters: {filters}
            
            DATABASE SCHEMA:
            {schema}
            
            Return ONLY a JSON object in this exact format:
            {{
                "query": "SELECT ... FROM ... WHERE ...",
                "explanation": "Brief explanation of the query",
                "expected_output": "Description of result columns"
            }}
            
            Example for finding top 10 actors with most movies:
            {{
                "query": "SELECT a.first_name, a.last_name, COUNT(fa.film_id) as movie_count FROM actor a JOIN film_actor fa ON a.actor_id = fa.actor_id GROUP BY a.actor_id, a.first_name, a.last_name ORDER BY movie_count DESC LIMIT 10",
                "explanation": "Joins actor and film_actor tables, counts films per actor, orders by count descending",
                "expected_output": "First name, last name, and count of movies for each actor"
            }}

            IMPORTANT: Return ONLY the JSON object, no other text or explanations.
            """)
            
            # Get LLM response
            response = llm.invoke(
                prompt.format(
                    intent=analysis.get('intent', ''),
                    tables=json.dumps(analysis.get('required_tables', [])),
                    columns=json.dumps(analysis.get('required_columns', [])),
                    aggregations=json.dumps(analysis.get('aggregations', [])),
                    filters=json.dumps(analysis.get('filters', [])),
                    schema=json.dumps(schema, indent=2)
                )
            )
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            logger.info("Got LLM response for SQL generation")
            logger.debug(f"Raw response: {response_text}")
            
            try:
                # Clean and parse the response
                cleaned_text = response_text.strip()
                
                # Remove any markdown or extra text
                if "{" in cleaned_text:
                    start_idx = cleaned_text.find("{")
                    end_idx = cleaned_text.rfind("}")
                    if end_idx == -1:  # No closing brace found
                        cleaned_text = cleaned_text[start_idx:] + "}"
                    else:
                        cleaned_text = cleaned_text[start_idx:end_idx + 1]
                
                # Ensure valid JSON structure
                try:
                    # First try to parse as is
                    json_response = json.loads(cleaned_text)
                except json.JSONDecodeError:
                    # If that fails, try to fix common formatting issues
                    cleaned_text = cleaned_text.replace('\n', '')
                    cleaned_text = cleaned_text.replace('  ', '')
                    if not cleaned_text.endswith('}'):
                        cleaned_text += '}'
                    json_response = json.loads(cleaned_text)
                
                # Validate required fields
                required_fields = ["query", "explanation", "expected_output"]
                for field in required_fields:
                    if field not in json_response:
                        raise ValueError(f"Missing required field: {field}")
                
                # Create SQLGeneration object
                sql_response = SQLGeneration(
                    query=json_response["query"],
                    explanation=json_response["explanation"],
                    expected_output=json_response["expected_output"]
                )
                
                logger.info(f"Successfully generated SQL: {sql_response.dict()}")
                return {"sql_generation": sql_response.dict()}
                
            except Exception as e:
                logger.error(f"Failed to parse SQL generation: {str(e)}")
                logger.error(f"Raw response was: {response_text}")
                return {
                    "error": "Failed to generate SQL",
                    "suggestion": "Please ensure your question is clear",
                    "details": str(e)
                }
                
        except Exception as e:
            logger.error(f"SQL generation failed: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def validate_sql(state: AnalystState) -> Dict:
        """Validate the generated SQL query."""
        try:
            logger.info("Starting SQL validation step")
            sql_gen = state.get('sql_generation', {})
            query = sql_gen.get('query', '')
            
            logger.info(f"Validating SQL: {json.dumps(sql_gen, indent=2)}")
            
            if not query:
                return {
                    "error": "No SQL query to validate",
                    "details": "SQL generation step did not produce a query"
                }

            prompt = PromptTemplate.from_template("""
            Validate this SQL query and return ONLY a JSON response:
            
            QUERY: {query}
            
            DATABASE SCHEMA:
            {schema}
            
            Return a JSON object in this exact format:
            {{
                "is_valid": true or false,
                "issues": ["list of issues if any"],
                "fixed_query": "the original query or fixed version if needed",
                "table_dependencies": ["list of tables used"],
                "column_dependencies": ["list of columns used"]
            }}
            
            Example response for a valid query:
            {{
                "is_valid": true,
                "issues": [],
                "fixed_query": "SELECT a.first_name, a.last_name, COUNT(fa.film_id) as movie_count FROM actor a JOIN film_actor fa ON a.actor_id = fa.actor_id GROUP BY a.actor_id, a.first_name, a.last_name ORDER BY movie_count DESC LIMIT 10",
                "table_dependencies": ["actor", "film_actor"],
                "column_dependencies": ["actor.first_name", "actor.last_name", "film_actor.film_id", "actor.actor_id"]
            }}
            
            IMPORTANT: Return ONLY the JSON object, no other text or code.
            """)

            # Get validation response
            response = llm.invoke(
                prompt.format(
                    query=query,
                    schema=json.dumps(db_connection.get_schema(), indent=2)
                )
            )
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"Raw validation response: {response_text}")
            
            try:
                # Clean and parse the response
                cleaned_text = response_text.strip()
                if "{" in cleaned_text:
                    start_idx = cleaned_text.find("{")
                    end_idx = cleaned_text.rfind("}")
                    if end_idx == -1:
                        cleaned_text = cleaned_text[start_idx:] + "}"
                    else:
                        cleaned_text = cleaned_text[start_idx:end_idx + 1]
                
                # Parse JSON
                validation = json.loads(cleaned_text)
                
                # Create SQLValidation object
                validation_result = SQLValidation(
                    is_valid=validation["is_valid"],
                    issues=validation["issues"],
                    fixed_query=validation["fixed_query"],
                    table_dependencies=validation["table_dependencies"],
                    column_dependencies=validation["column_dependencies"]
                )
                
                logger.info(f"Validation completed: {validation_result.dict()}")
                return {"sql_validation": validation_result.dict()}
                
            except Exception as e:
                logger.error(f"Invalid JSON response from validator: {response_text}")
                return {
                    "error": "SQL validation failed",
                    "details": str(e)
                }
                
        except Exception as e:
            logger.error(f"Error in SQL validation: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def execute_query(state: AnalystState) -> Dict:
        """Execute SQL query using the database connection."""
        try:
            logger.info("Starting query execution step")
            validation = state.get('sql_validation', {})
            
            if not validation or 'error' in validation:
                return {
                    "error": "Cannot execute query: Invalid SQL validation",
                    "details": validation.get('error', 'No valid SQL available')
                }
            
            query = validation.get('fixed_query', '')
            if not query:
                return {
                    "error": "No SQL query to execute",
                    "details": "SQL validation did not provide a query"
                }
            
            try:
                start_time = time.time()
                # Use the existing db_connection instance
                results = db_connection.execute_query(query)
                execution_time = time.time() - start_time
                
                logger.debug(f"Raw query results shape: {results.shape}")
                
                # Convert results to the expected format
                data = results.to_dict('records')
                headers = list(results.columns)
                
                logger.info(f"Query executed in {execution_time:.2f}s, returned {len(data)} rows")
                return {
                    "query_results": {
                        "data": data,
                        "column_names": headers,
                        "row_count": len(data),
                        "execution_time": execution_time
                    }
                }
                
            except Exception as exec_error:
                logger.error(f"Query execution error: {str(exec_error)}")
                logger.error(f"Failed query: {query}")
                return {
                    "error": "Query execution failed",
                    "details": str(exec_error)
                }
                
        except Exception as e:
            logger.error(f"Error in query execution: {str(e)}", exc_info=True)
            return {
                "error": "Query processing failed",
                "details": str(e)
            }

    def design_visualization(state: AnalystState) -> Dict:
        """Design and create visualization using Plotly."""
        try:
            results = state.get('query_results', {})
            analysis = state.get('question_analysis', {})
            
            if "error" in results:
                return {
                    "visualization_spec": {
                        "error": "Cannot create visualization: invalid query results",
                        "details": results.get("error")
                    }
                }
            
            data = results.get('data', [])
            if not data:
                return {
                    "visualization_spec": {
                        "error": "No data to visualize",
                        "details": "Query returned empty results"
                    }
                }
            
            # Convert data to DataFrame for easier handling
            df = pd.DataFrame(data)
            
            # Create Plotly visualization code
            viz_code = f"""
import plotly.graph_objects as go
import pandas as pd

# Create figure
fig = go.Figure()

# Add bars
fig.add_trace(go.Bar(
    x=[row['first_name'] + ' ' + row['last_name'] for row in {data}],
    y=[row['movie_count'] for row in {data}],
    name='Movies per Actor'
))

# Update layout
fig.update_layout(
    title='{analysis.get("intent", "Actor Movie Counts")}',
    xaxis_title='Actor Name',
    yaxis_title='Number of Movies',
    showlegend=True,
    template='plotly_white'
)

# Show figure
fig.show()
"""
            
            # Execute the visualization code
            try:
                python_repl = PythonREPL()
                viz_result = python_repl.run(viz_code)
                
                # Create visualization spec
                viz_spec = {
                    "type": "bar",
                    "x_axis": "actor_name",
                    "y_axis": "movie_count",
                    "title": analysis.get("intent", "Actor Movie Counts"),
                    "plotly_code": viz_code,
                    "execution_result": viz_result
                }
                
                logger.info("Successfully created visualization")
                return {"visualization_spec": viz_spec}
                
            except Exception as viz_error:
                logger.error(f"Visualization execution error: {str(viz_error)}")
                return {
                    "visualization_spec": {
                        "error": "Failed to create visualization",
                        "details": str(viz_error),
                        "plotly_code": viz_code
                    }
                }
                
        except Exception as e:
            logger.error(f"Error in visualization design: {str(e)}")
            return {
                "visualization_spec": {
                    "error": "Failed to design visualization",
                    "details": str(e)
                }
            }

    def format_final_output(state: AnalystState) -> Dict:
        """Format the final output with visualization."""
        try:
            viz_spec = state['visualization_spec']
            results = state['query_results']
            
            # Create final output with visualization
            final_output = {
                "data": results,
                "visualization": viz_spec,
                "summary": "Analysis complete"  # Add proper summary
            }
            
            return {"final_output": final_output}
            
        except Exception as e:
            logger.error(f"Error in output formatting: {str(e)}")
            raise

    def should_continue(state: AnalystState) -> str:
        """Route to next step or end on error."""
        current_step = state.get("current_step", "unknown")
        if "error" in state:
            error_msg = state["error"]
            logger.error(f"Error in {current_step}: {error_msg}")
            
            # Create a serializable version of the state
            serializable_state = {
                "messages": [msg.content for msg in state.get("messages", [])],
                "error": state.get("error"),
                "current_step": current_step,
                "question_analysis": state.get("question_analysis", {}),
                "sql_generation": state.get("sql_generation", {}),
                "sql_validation": state.get("sql_validation", {}),
                "query_results": state.get("query_results", {})
            }
            logger.error(f"State at failure: {json.dumps(serializable_state, indent=2)}")
            return END
            
        logger.info(f"Successfully completed {current_step}, continuing workflow")
        return "continue"

    # Build the graph with logging
    logger.info("Building analysis workflow graph")
    workflow = StateGraph(AnalystState)
    
    # Add nodes with logging
    logger.info("Adding workflow nodes")
    workflow.add_node("question_analyzer", analyze_question)
    workflow.add_node("sql_generator", generate_sql)
    workflow.add_node("sql_validator", validate_sql)
    workflow.add_node("query_executor", execute_query)
    workflow.add_node("visualization_designer", design_visualization)
    workflow.add_node("output_formatter", format_final_output)

    # Add edges with logging
    logger.info("Configuring workflow edges")
    workflow.add_conditional_edges(
        "question_analyzer",
        should_continue,
        {
            "continue": "sql_generator",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "sql_generator",
        should_continue,
        {
            "continue": "sql_validator",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "sql_validator",
        should_continue,
        {
            "continue": "query_executor",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "query_executor",
        should_continue,
        {
            "continue": "visualization_designer",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "visualization_designer",
        should_continue,
        {
            "continue": "output_formatter",
            END: END
        }
    )
    
    workflow.add_edge("output_formatter", END)

    # Set up the entry point
    workflow.add_edge(START, "question_analyzer")

    # Create the checkpointer with dedicated database
    checkpointer = SqliteSaver(
        connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    )

    logger.info("Workflow graph built successfully")
    return workflow.compile(checkpointer=checkpointer)
