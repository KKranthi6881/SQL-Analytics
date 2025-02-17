from typing import Dict, List, Any, Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.graph import Graph, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import uuid
from logging import Logger

from src.tools import SearchTools

# Define state types
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    code_context: Annotated[Dict, "Code search results"]
    doc_context: Annotated[Dict, "Documentation search results"]
    summary: Annotated[str, "Final summarized response"]

def create_analysis_system(tools: SearchTools):
    # Initialize models with specific configurations
    base_config = {
        "temperature": 0,
        "base_url": "http://localhost:11434",  # Ensure this matches your Ollama setup
        "timeout": 120,  # Increased timeout for longer operations
    }
    
    code_model = ChatOllama(
        model="llama3.2:3b",  
        **base_config
    )
    doc_model = ChatOllama(
        model="llama3.2:3b",
        **base_config
    )
    summary_model = ChatOllama(
        model="llama3.2:3b",
        **base_config
    )

    # Create specialized agents with more specific prompts
    code_agent = create_react_agent(
        model=code_model,
        tools=[tools.search_code],
        name="code_expert",
        prompt="""You are an expert code analyzer focusing only on the provided codebase. 
        
        Guidelines:
        - You have to provide only tables,columns and realtionship info only.
        - Focus more on how to build the right business with objects for other agents to build a
        - Only analyze code from the search results
        - Focus on technical implementation details
        - Identify patterns and relationships in the code
        - Do not make assumptions about code you cannot see
        - Always use the search tools before answering
        
        Current context will be provided through the search tools."""
    )

    doc_agent = create_react_agent(
        model=doc_model,
        tools=[tools.search_documentation, tools.search_relationships],
        name="documentation_expert",
        prompt="""You are a documentation expert focusing only on the provided documentation.
        
        Guidelines:
        - Only reference documentation from the search results
        - Focus on business requirements and workflows
        - Identify relationships between components
        - Do not make assumptions about documentation you cannot see
        - Always use the search tools before answering
        
        Current context will be provided through the search tools."""
    )

    summary_agent = create_react_agent(
        model=summary_model,
        tools=[tools.search_relationships],  # Give it ability to check relationships
        name="summary_expert",
        prompt="""You are an expert at analyzing and summarizing code and documentation relationships.
        
        Guidelines:
        - Focus on table structures, columns, and their relationships
        - Identify business objects and their technical implementations
        - Explain how different components connect and interact
        - Only reference information from the provided context
        - Structure your response in a clear, technical format
        
        Format your response as:
        1. Tables and Columns:
           - List main tables with their key columns
           - Highlight primary and foreign keys
        
        2. Relationships:
           - Show how tables are connected
           - Explain join conditions and cardinality
        
        3. Business Objects:
           - Map tables to business concepts
           - Explain how they work together
        
        4. Implementation Details:
           - Note any specific SQL patterns
           - Highlight performance considerations
        """
    )

    def supervisor_node(state: AgentState) -> AgentState:
        """Route and process the query through appropriate agents."""
        try:
            messages = state['messages']
            if not messages:
                return state
            
            # Convert dict message to BaseMessage if needed
            last_message = messages[-1]
            if isinstance(last_message, dict):
                query = last_message.get('content', '')
                last_message = HumanMessage(content=query)
            elif isinstance(last_message, BaseMessage):
                query = last_message.content
            else:
                query = str(last_message)

            # Process with both agents
            code_result = code_agent.invoke({
                "messages": [HumanMessage(content=query)],
                "code_context": state.get('code_context', {}),
                "doc_context": state.get('doc_context', {})
            })
            
            doc_result = doc_agent.invoke({
                "messages": [HumanMessage(content=query)],
                "code_context": state.get('code_context', {}),
                "doc_context": state.get('doc_context', {})
            })

            # Extract and format the results
            state['code_context'] = {
                "query": query,
                "results": code_result.get('code_context', {}).get('results', [])
            }
            state['doc_context'] = {
                "query": query,
                "results": doc_result.get('doc_context', {}).get('results', [])
            }
            
            return state
            
        except Exception as e:
            Logger.error(f"Error in supervisor node: {str(e)}")
            state['error'] = str(e)
            return state

    def summarize(state: AgentState) -> AgentState:
        """Use React agent to analyze and summarize the findings."""
        try:
            # Format the context for the summary agent
            code_results = state.get('code_context', {}).get('results', [])
            doc_results = state.get('doc_context', {}).get('results', [])
            
            # Create a structured context message
            context_message = {
                "code_analysis": [],
                "documentation": [],
                "relationships": []
            }

            # Process code results
            if isinstance(code_results, list):
                for result in code_results:
                    if isinstance(result, dict):
                        code_entry = {
                            "code": result.get('code', ''),
                            "file_info": result.get('file_info', {}),
                            "type": "sql" if ".sql" in str(result.get('file_info', {}).get('file_path', '')) else "python"
                        }
                        context_message["code_analysis"].append(code_entry)

            # Process documentation results
            if isinstance(doc_results, list):
                for result in doc_results:
                    if isinstance(result, dict):
                        doc_entry = {
                            "content": result.get('content', ''),
                            "metadata": result.get('metadata', {})
                        }
                        context_message["documentation"].append(doc_entry)

            # Create a structured prompt for the summary agent
            analysis_prompt = f"""
            Analyze and summarize the following codebase information:

            Code Analysis:
            {context_message['code_analysis']}

            Documentation:
            {context_message['documentation']}

            Provide a technical summary focusing on:
            1. Table structures and relationships
            2. Business object implementations
            3. Data flow patterns
            4. Key technical considerations
            """

            # Get summary from the React agent
            summary_result = summary_agent.invoke({
                "messages": [HumanMessage(content=analysis_prompt)],
                "context": context_message
            })

            # Extract and format the summary
            if isinstance(summary_result, dict):
                state['summary'] = summary_result.get('output', summary_result.get('response', 'No summary available'))
            else:
                state['summary'] = str(summary_result)

            return state
            
        except Exception as e:
            Logger.error(f"Error generating summary: {str(e)}")
            state['summary'] = f"Error generating summary: {str(e)}"
            return state

    # Build the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("summarizer", summarize)

    # Add edges
    graph.add_edge('supervisor', 'summarizer')
    graph.set_entry_point("supervisor")
    graph.set_finish_point("summarizer")

    return graph.compile(checkpointer=MemorySaver())

class AnalysisSystem:
    def __init__(self, tools: SearchTools):
        self.app = create_analysis_system(tools)

    def analyze(self, query: str) -> Dict[str, Any]:
        """Process a query through the analysis system."""
        try:
            result = self.app.stream({
                "messages": [HumanMessage(content=query)],
                "code_context": {},
                "doc_context": {},
                "summary": ""
            }, 
            {"configurable": {"thread_id": str(uuid.uuid4())}}
            )
            
            return {
                "summary": result.get("summary", "No summary available"),
                "code_context": result.get("code_context", {}),
                "doc_context": result.get("doc_context", {})
            }
            
        except Exception as e:
            return {
                "summary": f"Error during analysis: {str(e)}",
                "code_context": {},
                "doc_context": {}
            }





