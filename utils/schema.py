from typing import List
from pydantic import BaseModel, Field
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
import os
from dotenv import load_dotenv
import datetime

# Load environment variables from .env file
load_dotenv()

# Basic document structure
class Document(BaseModel):
    content: str = Field(..., description="The content of the document")
    metadata: dict = Field(
        default_factory=dict, description="Document metadata"
    )

# Structure for query responses
class QueryResult(BaseModel):
    answer: str = Field(..., description="Response to the query")
    source_nodes: List[str] = Field(
        ..., description="Source references used"
    )

class StockAssistant:
    def __init__(
        self,
        data_path: str,
        index_path: str = "index",
    ):
        """
        Initialize the AI Assistant
        :param data_path: Path to your document directory
        :param index_path: Path where the vector index will be stored
        """
        self.data_path = data_path
        self.index_path = index_path
        
        # Customize this prompt for your use case
        self.system_prompt = """
        You are an AI Assistant that helps users understand and retrieve information related to the basics of stocks and investments. Your responses are based solely on a comprehensive document covering topics such as investments, stock markets, IPOs, corporate actions, and market regulators. Here’s how you should approach your tasks:

        Primary Role
        Your main responsibility is to help users by extracting relevant information from the document, which covers investment basics, financial instruments, the stock market, IPOs, and various corporate actions.
        Provide users with clear and detailed information on concepts, terms, and processes related to investing, stock trading, and market regulation.
        Topics to Cover
        Summarize information on investment instruments (e.g., equities, commodities, real estate, fixed income) and their characteristics.
        Explain stock market fundamentals, including the roles of regulators, market participants, and financial intermediaries.
        Describe the IPO process, including why companies go public and the sequence of events involved.
        Clarify corporate actions (such as dividends, stock splits, buybacks) and their impact on stock prices.
        Provide guidance on trading platforms, understanding market indices, and calculating investment returns.
        Offer insights on key events (like inflation, monetary policy, or corporate earnings) that can influence the stock market.
        Information Extraction and Response Style
        Provide concise and informative answers based directly on the document’s content.
        Where necessary, include quotes or references from the document to support your answer.
        If a query pertains to a specific section or chapter, explain related concepts and terms in a simple and understandable way.
        If the user requests definitions or explanations, share the document’s exact wording first, followed by a simple paraphrase if needed.
        Handling General Questions
        If a query mentions recent events or current market conditions not covered in the document, explain that the document may not cover events beyond its publication date and suggest consulting recent sources if necessary.
        For detailed or complex queries, break down the response into sections and refer to relevant chapters or document sections as needed.
        Clarity and Usability
        Aim to make each response easy to understand, accurate, and informative, using the document as your only source of information.
        """

        self.configure_settings()
        self.index = None
        self.agent = None
        self.load_or_create_index()

    def configure_settings(self):
        """Configure LLM and embedding settings"""
        # Replace with your preferred LLM
        Settings.llm = Groq(model="mixtral-8x7b-32768", api_key=os.getenv("GROQ_API_KEY"))  # Add your LLM configuration here
        # Replace with your preferred embedding model
        Settings.embed_model = JinaEmbedding(
            api_key=os.getenv("JINA_API_KEY"),
            model="jina-embeddings-v3",
            task='retrieval.passage'
        )  # Add your embedding model configuration here

    def load_or_create_index(self):
        """Load existing index or create new one"""
        if os.path.exists(self.index_path):
            self.load_index()
        else:
            self.create_index()
        self._create_agent()

    def load_index(self):
        """Load existing vector index"""
        storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
        self.index = load_index_from_storage(storage_context)

    def create_index(self):
        """Create new vector index from documents"""
        documents = SimpleDirectoryReader(
            self.data_path,
            recursive=True,
        ).load_data()
        if not documents:
            raise ValueError("No documents found in specified path")
        self.index = VectorStoreIndex.from_documents(documents)
        self.save_index()

    def _create_agent(self):
        """Set up the agent with custom tools"""
        query_engine = self.index.as_query_engine(similarity_top_k=5)
        
        # Basic search tool
        search_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="document_search",
                description="Search through the document database",
            ),
        )

        # Initialize the agent with tools
        self.agent = ReActAgent.from_tools(
            [search_tool],
            verbose=True,
            system_prompt=self.system_prompt,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
        )

    def query(self, query: str) -> QueryResult:
        """
        Process a query and return results
        :param query: User's question or request
        :return: QueryResult with answer and sources
        """
        if not self.agent:
            raise ValueError("Agent not initialized")
        response = self.agent.chat(query)
        return QueryResult(
            answer=response.response,
            source_nodes=[],
        )

    def save_index(self):
        """Save the vector index to disk"""
        os.makedirs(self.index_path, exist_ok=True)
        self.index.storage_context.persist(persist_dir=self.index_path)

# Example usage
if __name__ == "__main__":
    assistant = StockAssistant(
        data_path="./data",
        index_path="./index"
    )
    result = assistant.query("Your question here")
    print(result.answer)