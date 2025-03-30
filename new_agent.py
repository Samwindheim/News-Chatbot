"""
News Chatbot: A conversational agent for retrieved news articles

This program provides a chat interface to interact with news articles
that have been scraped and stored in a vector database. It uses
LangChain to retrieve relevant articles based on user queries and
maintains conversation history for context-aware responses.

Key Features:
- Vector-based retrieval of relevant news articles
- Conversation memory for contextual responses
- LLM-powered answer generation with source attribution
- Interactive chat interface
"""

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env
from web_scraper import run_scraper

class NewsChatbot:
    def __init__(self, chroma_directory="./chroma_db", refresh_data=False):
        # Initialize the language model
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        
        # Refresh data if requested
        if refresh_data:
            asyncio.run(run_scraper(reset_db=True))
        
        # Connect to the vector database
        self.chroma_db = Chroma(persist_directory=chroma_directory)
        
        # Set up the workflow with memory
        self.workflow = StateGraph(state_schema=MessagesState)
        self.app = self.setup_chat_workflow()
        
    def create_augmented_response(self, state: MessagesState):
        """
        Generate responses using conversation history and relevant news articles.
        """
        # Get the latest question
        latest_msg = state["messages"][-1].content if state["messages"] else ""
        
        # Search for relevant articles using MMR for diversity
        results = self.chroma_db.max_marginal_relevance_search(
            query=latest_msg,
            k=3,               # Number of articles to return
            fetch_k=5,         # Initial pool to fetch
            lambda_mult=0.7    # Balance between relevance and diversity
        )
        
        # Format the context from retrieved articles
        context_parts = []
        for i, doc in enumerate(results, 1):
            metadata = doc.metadata
            title = metadata.get("title", "No title")
            source = metadata.get("source", "Unknown source")
            date = metadata.get("date", "Unknown date")
            
            context_parts.append(
                f"ARTICLE {i}:\nTitle: {title}\nSource: {source}\nDate: {date}\n"
                f"Content: {doc.page_content}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Create system message with context
        system_msg = SystemMessage(content=(
            "You are a helpful news assistant. Use both the conversation history "
            "and the provided news articles to give accurate, informative answers. "
            "If the articles don't contain relevant information, say so honestly. "
            "Always cite your sources by referring to the article number.\n\n"
            f"ARTICLES:\n{context}"
        ))
        
        # Combine system message with conversation history
        messages = [system_msg] + state["messages"]
        
        # Get response from the model
        response = self.llm.invoke(messages)
        return {"messages": response}
    
    def setup_chat_workflow(self):
        """Create and configure the chat workflow with memory management."""
        # Define the function that processes messages
        def process_messages(state: MessagesState):
            return self.create_augmented_response(state)
        
        # Add node and edge to workflow
        self.workflow.add_node("chat", process_messages)
        self.workflow.add_edge(START, "chat")
        
        # Add memory management
        memory = MemorySaver()
        return self.workflow.compile(checkpointer=memory)
    
    def chat(self, question: str, thread_id: str = "default"):
        """Process a user message while maintaining conversation history."""
        response = self.app.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        # Return just the latest AI response
        return response["messages"][-1].content
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        thread_id = f"user_session_{os.getpid()}"
        print("News Chatbot: Ask me about recent news articles! (Press Ctrl+C to exit)")
        
        try:
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("News Chatbot: Goodbye!")
                    break
                    
                response = self.chat(user_input, thread_id)
                print(f"\nNews Chatbot: {response}")
                
        except KeyboardInterrupt:
            print("\nExiting chat. Goodbye!")

if __name__ == "__main__":
    # Initialize chatbot (set refresh_data=True to scrape fresh data)
    chatbot = NewsChatbot(refresh_data=False)
    
    # Start interactive chat
    chatbot.interactive_chat()