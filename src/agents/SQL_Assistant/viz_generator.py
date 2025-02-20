from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Visualization:
    """Structure to hold visualization details"""
    plot_type: str
    code: str
    explanation: str
    title: str
    x_axis: str
    y_axis: str
    additional_params: Dict[str, Any]

class VizGenerator:
    """Generates visualizations from SQL query results"""
    
    def __init__(self):
        self.setup_components()
        self.python_repl = PythonREPLTool()

    def setup_components(self):
        """Initialize LLM and prompt"""
        self.llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0,
            base_url="http://localhost:11434"
        )
        
        self.viz_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data visualization engineer creating business-ready visualizations. Create clear, professional, and insightful visualizations.

            Important Guidelines:
            1. Choose visualization type based on:
               - Data structure and relationships
               - Question being answered
               - Business audience needs
               - Data storytelling principles
            
            2. Chart Enhancement Requirements:
               - Professional color schemes
               - Meaningful tooltips with formatted values
               - Currency symbols where applicable
               - Formatted numbers (e.g., 1M)
               - Descriptive titles and subtitles
               - Clear axis labels
               - Chart size (height=600, width=1000)
               - Interactive hover effects
            
            3. Styling Requirements:
               - Consistent layout styling
               - Gridlines for readability
               - Professional fonts (Arial)
               - Adequate spacing
               - Data labels where appropriate
            
            4. Business Focus:
               - Key insights highlighted
               - Clear trends
               - Business-friendly formatting
               - Color-blind friendly

            For geographical data, use:
            - Choropleth maps (px.choropleth)
            - Scatter geo plots (px.scatter_geo)
            - Bubble maps (px.scatter_geo with size parameter)

            For categorical data, use:
            - Bar charts (px.bar)
            - Treemaps (px.treemap)
            - Sunburst charts (px.sunburst)

            For temporal data, use:
            - Line charts (px.line)
            - Area charts (px.area)
            - Timeline plots (px.timeline)

            Available Data:
            - Columns: {columns}
            - Sample: {data_sample}
            - Shape: {data_shape}
            
            Example Codes:

            1. Bar Chart:
            ```python
            fig = px.bar(df, 
                x='category', 
                y='value',
                title='Business Metrics',
                color='category',
                height=600,
                width=1000
            )
            fig.update_layout(
                template='plotly_white',
                font=dict(family='Arial', size=12),
                showlegend=True
            )
            ```

            2. World Map:
            ```python
            fig = px.choropleth(df,
                locations='country',
                locationmode='country names',
                color='value',
                title='Global Distribution',
                height=600,
                width=1000,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                template='plotly_white',
                font=dict(family='Arial', size=12)
            )
            ```

            3. Time Series:
            ```python
            fig = px.line(df,
                x='date',
                y='value',
                title='Trend Analysis',
                height=600,
                width=1000
            )
            fig.update_layout(
                template='plotly_white',
                font=dict(family='Arial', size=12)
            )
            ```
            """),
            
            ("human", """Question: {question}
            Data Description: {data_description}
            
            Generate a business-ready visualization that best answers this question.
            
            Respond with:
            ANALYSIS: [explain the business insights]
            
            VISUALIZATION_TYPE: [chart type with justification]
            
            CODE:
            [complete plotly code with styling]
            
            EXPLANATION: [explain visualization choices]
            """)
        ])

    def generate_visualization(
        self,
        df: pd.DataFrame,
        question: str,
        data_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate visualization for the query results"""
        try:
            # Log the data we're working with
            logger.info(f"Generating visualization for question: {question}")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
            logger.info(f"Data sample:\n{df.head()}")
            
            # Get visualization suggestion from LLM
            response = self.llm.invoke(
                self.viz_prompt.format(
                    columns=df.columns.tolist(),
                    data_sample=df.head().to_dict('records'),
                    data_shape=df.shape,
                    question=question,
                    data_description=(
                        data_description or 
                        f"DataFrame with {len(df)} rows and columns: {', '.join(df.columns)}"
                    )
                )
            )
            
            # Parse and clean the response
            viz_details = self._parse_viz_response(response.content)
            clean_code = viz_details.code.replace('```python', '').replace('```', '').strip()
            
            logger.info(f"Executing visualization code:\n{clean_code}")
            
            # Execute the visualization code
            try:
                namespace = {
                    'pd': pd,
                    'px': px,
                    'go': go,
                    'df': df,
                    'fig': None,
                    'np': np,
                    'colors': px.colors
                }
                
                # Execute the code
                exec(clean_code, namespace)
                
                # Get the figure
                fig = namespace.get('fig')
                if fig is None:
                    logger.warning("No figure object created, checking for last expression")
                    last_line = clean_code.strip().split('\n')[-1]
                    if not last_line.startswith(('fig.', 'fig=')):
                        logger.info("Adding fig assignment to last expression")
                        clean_code += '\nfig = ' + last_line
                        exec(clean_code, namespace)
                        fig = namespace.get('fig')
                
                if fig is None:
                    raise ValueError("No visualization figure was created")
                
                # Update layout with consistent styling
                fig.update_layout(
                    template='plotly_white',
                    font=dict(family='Arial', size=12),
                    margin=dict(t=100, l=50, r=50, b=50),
                    hoverlabel=dict(font_size=12, font_family='Arial'),
                    showlegend=True
                )
                
                return {
                    "status": "success",
                    "result": viz_details,
                    "figure": fig
                }
                
            except Exception as e:
                logger.error(f"Visualization execution failed: {str(e)}", exc_info=True)
                return {"status": "error", "error": str(e)}
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}", exc_info=True)
            return {"status": "error", "error": str(e)}

    def _parse_viz_response(self, response: str) -> Visualization:
        """Parse LLM response into visualization details"""
        plot_type = ""
        code = ""
        explanation = ""
        title = ""
        x_axis = ""
        y_axis = ""
        additional_params = {}
        
        current_section = None
        current_content = []
        
        for line in response.split('\n'):
            line = line.strip()
            if not line or line.startswith('```'):  # Skip empty lines and code markers
                continue
                
            if line.upper().startswith('ANALYSIS:'):
                current_section = 'analysis'
                current_content = []
            elif line.upper().startswith('VISUALIZATION_TYPE:'):
                if current_section == 'analysis':
                    explanation = '\n'.join(current_content).strip()
                plot_type = line.split(':', 1)[1].strip()
                current_section = None
            elif line.upper().startswith('CODE:'):
                current_section = 'code'
                current_content = []
            elif line.upper().startswith('EXPLANATION:'):
                if current_section == 'code':
                    code = '\n'.join(current_content).strip()
                current_section = 'explanation'
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Handle last section
        if current_section == 'explanation':
            explanation += '\n' + '\n'.join(current_content).strip()
        elif current_section == 'code':
            code = '\n'.join(current_content).strip()
        
        # Extract title and axis info from code
        for line in code.split('\n'):
            if 'title=' in line:
                title = line.split('title=')[1].split(',')[0].strip().strip('"\'')
            if 'x=' in line:
                x_axis = line.split('x=')[1].split(',')[0].strip().strip('"\'')
            if 'y=' in line:
                y_axis = line.split('y=')[1].split(',')[0].strip().strip('"\'')
        
        # If no title found in code, generate one from question
        if not title:
            title = f"Distribution of {y_axis} by {x_axis}" if x_axis and y_axis else "Analysis Results"
        
        logger.info(f"Parsed visualization details:")
        logger.info(f"Plot type: {plot_type}")
        logger.info(f"Title: {title}")
        logger.info(f"X-axis: {x_axis}")
        logger.info(f"Y-axis: {y_axis}")
        
        return Visualization(
            plot_type=plot_type,
            code=code,
            explanation=explanation,
            title=title,
            x_axis=x_axis,
            y_axis=y_axis,
            additional_params=additional_params
        )

# Example usage
if __name__ == "__main__":
    from src.agents.SQL_Assistant.sql_generator import SQLQueryGenerator
    from src.agents.SQL_Assistant.question_parser import QuestionParser
    from src.agents.SQL_Assistant.doc_search import SimpleDocSearch
    from src.agents.SQL_Assistant.code_search import SimpleCodeSearch
    from src.agents.SQL_Assistant.query_evaluator import SQLQueryEvaluator
    from src.agents.SQL_Assistant.query_executor import SQLQueryExecutor
    from src.tools import SearchTools
    from src.utils import ChromaDBManager
    import sqlite3
    
    # Setup components
    sqlite_path = "/Users/Kranthi_1/SQL-Analytics/src/db/sampledb/sakila_master.db"
    chroma_manager = ChromaDBManager()
    search_tools = SearchTools(chroma_manager)
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
    parser = QuestionParser(SimpleDocSearch(search_tools), SimpleCodeSearch(search_tools))
    sql_generator = SQLQueryGenerator()
    evaluator = SQLQueryEvaluator(db_schema)
    executor = SQLQueryExecutor(f"sqlite:///{sqlite_path}", evaluator)
    viz_generator = VizGenerator()
    
    # Test questions with expected visualization types
    test_cases = [
        {
            "question": "Which film categories have the highest rental frequency?  ",
            "expected_viz": "World map heatmap of customer distribution"
        }
    ]
    
    for test in test_cases:
        question = test["question"]
        print(f"\nProcessing: {question}")
        print(f"Expected visualization: {test['expected_viz']}")
        print("-" * 50)
        
        # Process pipeline
        parsed = parser.parse_question(question)
        if parsed["status"] == "success":
            generated = sql_generator.generate_query(parsed["result"], question)
            if generated["status"] == "success":
                result = executor.execute_query(generated["result"])
                
                if result["status"] == "success" and result["result"].success:
                    # Generate visualization
                    viz_result = viz_generator.generate_visualization(
                        df=result["result"].data,
                        question=question,
                        data_description=f"Query results with {len(result['result'].data)} rows"
                    )
                    
                    if viz_result["status"] == "success":
                        viz = viz_result["result"]
                        print("\nVisualization Details:")
                        print(f"Type: {viz.plot_type}")
                        print(f"Title: {viz.title}")
                        print("\nAnalysis:")
                        print(viz.explanation)
                        
                        # Display plot
                        fig = viz_result["figure"]
                        fig.show()
                    else:
                        print(f"Visualization error: {viz_result['error']}")
                else:
                    print(f"Query execution error: {result['error']}")
            else:
                print(f"SQL generation error: {generated['error']}")
        else:
            print(f"Question parsing error: {parsed['error']}") 