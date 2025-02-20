from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from src.agents.SQL_Assistant.sql_generator import GeneratedSQL
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryEvaluation:
    """Structure to hold query evaluation results"""
    is_valid: bool
    query: str
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    confidence: float

class SQLQueryEvaluator:
    """Evaluates SQL queries using LLM against schema, search context, and generated query"""
    
    def __init__(self, db_schema: Dict[str, Dict[str, Any]]):
        self.db_schema = db_schema
        self.setup_components()

    def setup_components(self):
        """Initialize LLM and prompt"""
        self.llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0,
            base_url="http://localhost:11434"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQLite validator. Analyze SQL queries for SQLite compatibility and correctness.
            
            Database Schema:
            {schema}
            
            Important SQLite Rules:
            1. No EXTRACT function - use strftime('%m', date_column) for month
            2. No native date functions - use strftime for date operations
            3. JOIN syntax must be explicit
            4. Column names must be fully qualified in JOINs
            5. Check all table/column names exist in schema
            
            Common SQLite Date Formats:
            - Extract Month: strftime('%m', date_column)
            - Extract Year: strftime('%Y', date_column)
            - Format Date: strftime('%Y-%m-%d', date_column)
            
            Focus on:
            1. SQLite syntax compatibility
            2. Table and column existence
            3. JOIN conditions correctness
            4. Date handling functions
            5. Aggregation logic
            
            Respond with:
            VALID: true/false
            ERRORS: [critical issues that must be fixed]
            WARNINGS: [potential problems]
            SUGGESTIONS: [improvements, especially for SQLite compatibility]
            CONFIDENCE: [0-1 score]
            CORRECTED_QUERY: [if needed, provide SQLite-compatible version]
            """),
            
            ("human", """User Question: {question}
            
            Generated SQL:
            {generated_sql}
            
            Business Context:
            {business_context}
            
            Evaluate if this query:
            1. Uses correct SQL syntax
            2. Properly handles dates/times
            3. Uses correct table relationships
            4. Answers the business question
            5. Needs any SQLite-specific adjustments
            6. You need to tweek the final output query for execution without syntax errors.""")
        ])

    def evaluate_query(
        self,
        query: str,
        question: str,
        business_context: Optional[Dict] = None,
        generated_sql: Optional[GeneratedSQL] = None
    ) -> Dict[str, Any]:
        """Evaluate SQL query against schema, context and generated query with double validation"""
        try:
            logger.info(f"Starting double validation for question: {question}")
            
            # First evaluation
            first_eval = self._perform_evaluation(
                query, question, business_context, generated_sql
            )
            logger.info("First evaluation complete")
            logger.info(f"First confidence score: {first_eval.confidence}")
            
            # Second evaluation with stricter criteria
            second_eval = self._perform_evaluation(
                query, question, business_context, generated_sql,
                strict_mode=True  # Enable stricter validation
            )
            logger.info("Second evaluation complete")
            logger.info(f"Second confidence score: {second_eval.confidence}")
            
            # Compare and merge evaluations
            final_evaluation = self._merge_evaluations(first_eval, second_eval)
            logger.info(f"Final merged confidence: {final_evaluation.confidence}")
            
            # Decide if query needs revision
            if final_evaluation.confidence < 0.7:  # Configurable threshold
                logger.warning("Low confidence score - query may need revision")
                final_evaluation.warnings.append(
                    "Low confidence in query - consider revision"
                )
            
            return {
                "status": "success",
                "result": final_evaluation
            }
            
        except Exception as e:
            logger.error(f"Query evaluation failed: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

    def _perform_evaluation(
        self,
        query: str,
        question: str,
        business_context: Optional[Dict],
        generated_sql: Optional[GeneratedSQL],
        strict_mode: bool = False
    ) -> QueryEvaluation:
        """Perform single evaluation pass"""
        
        # Add strict mode criteria to prompt if enabled
        extra_criteria = """
        Additional Validation Criteria:
        1. Double-check all table relationships
        2. Verify column data types
        3. Check for edge cases
        4. Validate business logic thoroughly
        5. Ensure optimal performance
        """ if strict_mode else ""
        
        context = self._format_business_context(business_context)
        sql_details = self._format_sql_details(generated_sql)
        
        response = self.llm.invoke(
            self.prompt.format(
                schema=self.db_schema,
                question=question,
                business_context=context,
                generated_sql=sql_details,
                extra_criteria=extra_criteria
            )
        )
        
        return self._parse_evaluation_response(response.content, query)

    def _merge_evaluations(
        self,
        first_eval: QueryEvaluation,
        second_eval: QueryEvaluation
    ) -> QueryEvaluation:
        """Merge two evaluations into final result"""
        
        # Combine unique errors and warnings
        all_errors = list(set(first_eval.errors + second_eval.errors))
        all_warnings = list(set(first_eval.warnings + second_eval.warnings))
        all_suggestions = list(set(first_eval.suggestions + second_eval.suggestions))
        
        # Calculate final confidence
        final_confidence = min(
            0.95,  # Cap maximum confidence
            (first_eval.confidence + second_eval.confidence) / 2
        )
        
        # Query is valid only if both evaluations agree
        is_valid = first_eval.is_valid and second_eval.is_valid
        
        return QueryEvaluation(
            is_valid=is_valid,
            query=first_eval.query,
            errors=all_errors,
            warnings=all_warnings,
            suggestions=all_suggestions,
            confidence=final_confidence
        )

    def _format_business_context(self, context: Optional[Dict]) -> str:
        """Format business context for prompt"""
        if not context:
            return "No business context provided"
        
        return f"""
        Key Concepts: {context.get('key_concepts', [])}
        Business Rules: {context.get('business_rules', [])}
        Analysis: {context.get('detailed_analysis', '')}
        """

    def _format_sql_details(self, sql: Optional[GeneratedSQL]) -> str:
        """Format SQL details for prompt"""
        if not sql:
            return "No SQL details provided"
        
        return f"""
        Original Query: {sql.query}
        Explanation: {sql.explanation}
        Tables Used: {sql.tables_used}
        Warnings: {sql.warnings}
        """

    def _parse_evaluation_response(self, response: str, original_query: str) -> QueryEvaluation:
        """Parse LLM response into evaluation result"""
        is_valid = False
        errors = []
        warnings = []
        suggestions = []
        confidence = 0.5
        
        for line in response.split('\n'):
            line = line.strip()
            if line.upper().startswith('VALID:'):
                is_valid = line.split(':', 1)[1].strip().lower() == 'true'
            elif line.upper().startswith('ERRORS:'):
                current_section = 'errors'
            elif line.upper().startswith('WARNINGS:'):
                current_section = 'warnings'
            elif line.upper().startswith('SUGGESTIONS:'):
                current_section = 'suggestions'
            elif line.upper().startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':')[1].strip())
                except:
                    confidence = 0.5
            elif line.startswith('-') and current_section:
                if current_section == 'errors':
                    errors.append(line[1:].strip())
                elif current_section == 'warnings':
                    warnings.append(line[1:].strip())
                elif current_section == 'suggestions':
                    suggestions.append(line[1:].strip())
        
        return QueryEvaluation(
            is_valid=is_valid,
            query=original_query,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            confidence=confidence
        )

# Example usage
if __name__ == "__main__":
    import sqlite3
    from src.agents.SQL_Assistant.sql_generator import SQLQueryGenerator
    from src.agents.SQL_Assistant.question_parser import QuestionParser
    from src.agents.SQL_Assistant.doc_search import SimpleDocSearch
    from src.agents.SQL_Assistant.code_search import SimpleCodeSearch
    from src.tools import SearchTools
    from src.utils import ChromaDBManager
    
    # Get schema from SQLite database
    sqlite_path = "/Users/Kranthi_1/SQL-Analytics/src/db/sampledb/sakila_master.db"
    
    def get_db_schema():
        """Extract schema from SQLite database"""
        schema = {}
        with sqlite3.connect(sqlite_path) as conn:
            # Get all tables
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            
            for table in tables:
                table_name = table[0]
                # Get columns
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
                    'columns': [col[1] for col in columns],
                    'relationships': relationships
                }
        return schema
    
    # Initialize components
    db_schema = get_db_schema()
    chroma_manager = ChromaDBManager()
    search_tools = SearchTools(chroma_manager)
    
    # Create parser
    parser = QuestionParser(
        SimpleDocSearch(search_tools),
        SimpleCodeSearch(search_tools)
    )
    
    # Create SQL generator and evaluator
    sql_generator = SQLQueryGenerator()
    evaluator = SQLQueryEvaluator(db_schema)
    
    # Test question
    question = "Show me the top 10 movies by rental count"
    
    # Process pipeline
    print(f"\nProcessing question: {question}")
    print("-" * 50)
    
    # Parse question
    parsed = parser.parse_question(question)
    if parsed["status"] == "success":
        # Generate SQL
        generated = sql_generator.generate_query(parsed["result"], question)
        
        if generated["status"] == "success":
            # Evaluate generated SQL
            result = evaluator.evaluate_query(
                query=generated["result"].query,
                question=question,
                business_context=parsed["result"].business_context,
                generated_sql=generated["result"]
            )
            
            if result["status"] == "success":
                eval_result = result["result"]
                print("\nGenerated SQL:")
                print(eval_result.query)
                
                print("\nValidation Results:")
                print(f"Valid: {eval_result.is_valid}")
                
                if eval_result.errors:
                    print("\nErrors:")
                    for error in eval_result.errors:
                        print(f"- {error}")
                
                if eval_result.warnings:
                    print("\nWarnings:")
                    for warning in eval_result.warnings:
                        print(f"- {warning}")
                
                if eval_result.suggestions:
                    print("\nSuggestions:")
                    for suggestion in eval_result.suggestions:
                        print(f"- {suggestion}")
                
                print(f"\nConfidence Score: {eval_result.confidence}")
            else:
                print(f"Evaluation failed: {result['error']}")
        else:
            print(f"SQL generation failed: {generated['error']}")
    else:
        print(f"Question parsing failed: {parsed['error']}") 