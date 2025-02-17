from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
from src.utils import ChromaDBManager

logger = logging.getLogger(__name__)

class CodeQAAgent:
    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager
        
        self.llm = ChatOllama(
            model="deepseek-r1:8b",
            temperature=0
        ) 
        
        '''
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        ) '''
        
        # Initialize prompt templates
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized code and documentation analysis assistant. Your task is to provide comprehensive answers using both code and documentation context.

            Guidelines:
            - Analyze both code and related documentation
            - Explain the business logic and technical implementation
            - Identify and explain table/column relationships
            - Provide data flow explanations
            - Connect technical details with business context
            - If information is missing from either code or docs, note it
            - you must diaply the summary of your think. don't give full context.
            
            Code Context:
            {code_context}
            
            Documentation Context:
            {doc_context}
            
            Table Relationships:
            {relationships}
            """),
            ("human", "{question}")
        ])

    def ask(self, question: str) -> Dict[str, Any]:
        """Process a question using both code and documentation context."""
        try:
            # Get all collections
            collections = self.chroma_manager.client.list_collections()
            if not collections:
                return {
                    "question": question,
                    "answer": "No collections found. Please upload code files and documentation first.",
                    "code_context": "No code available - please upload code files",
                    "doc_context": "No documentation available - please upload documentation",
                    "relationships": "No relationships available"
                }
            
            # Separate code and doc collections
            code_collections = [c.name for c in collections 
                              if any(ext in c.name.lower() 
                                    for ext in ['py', 'sql', 'yml', 'yaml'])]
            doc_collections = [c.name for c in collections 
                             if c.name not in code_collections]
            
            # Search code collections
            code_results = []
            for collection_name in code_collections:
                try:
                    results = self.chroma_manager.code_similarity_search(
                        collection_name=collection_name,
                        code_snippet=question,
                        n_results=2
                    )
                    if results and 'results' in results:
                        code_results.extend(results['results'])
                except Exception as e:
                    logger.warning(f"Error searching code collection {collection_name}: {str(e)}")
            
            # Search documentation collections
            doc_results = []
            for collection_name in doc_collections:
                try:
                    results = self.chroma_manager.hybrid_search(
                        collection_name=collection_name,
                        query=question,
                        n_results=2
                    )
                    if results and 'results' in results:
                        doc_results.extend(results['results'])
                except Exception as e:
                    logger.warning(f"Error searching doc collection {collection_name}: {str(e)}")
            
            # Sort results by similarity
            code_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            doc_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            # Format code context
            code_context_parts = []
            relationships_parts = []
            
            for result in code_results[:3]:
                # Add code content
                code_context_parts.append(f"Code Context:\n{result.get('code', '')}")
                
                # Add file info
                if 'file_info' in result:
                    code_context_parts.append(f"File: {result['file_info'].get('file_path', 'unknown')}")
                
                # Add matched lines
                if 'matched_lines' in result:
                    code_context_parts.append("Relevant Lines:")
                    for match in result['matched_lines']:
                        code_context_parts.append(f"{match['line_number']}: {match['content']}")
                
                # Extract relationships from metadata
                if 'file_info' in result and 'relationships' in result['file_info']:
                    relationships_parts.append(result['file_info']['relationships'])
            
            # Format documentation context
            doc_context_parts = []
            for result in doc_results[:3]:
                doc_context_parts.append(f"Documentation:\n{result.get('content', '')}")
                if 'metadata' in result:
                    doc_context_parts.append(f"Source: {result['metadata'].get('source', 'unknown')}")
            
            # Prepare context for LLM
            context = {
                "code_context": "\n\n".join(code_context_parts) if code_context_parts else "No relevant code found",
                "doc_context": "\n\n".join(doc_context_parts) if doc_context_parts else "No relevant documentation found",
                "relationships": "\n".join(relationships_parts) if relationships_parts else "No relationship information available",
                "question": question
            }
            
            # Generate answer
            answer = self.llm.invoke(
                self.qa_prompt.format(**context)
            )
            
            return {
                "question": question,
                "answer": answer.content if hasattr(answer, 'content') else str(answer),
                "code_context": context["code_context"],
                "doc_context": context["doc_context"],
                "relationships": context["relationships"]
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "question": question,
                "answer": f"Error processing your question: {str(e)}",
                "code_context": "Error occurred during processing",
                "doc_context": "No documentation available",
                "relationships": "No relationships available"
            } 