"""
News Scraper: A system for extracting structured data from The Guardian website

This program scrapes content from The Guardian, processes it using LLM extraction,
and stores the results in a Chroma vector database for later retrieval.
It handles the entire pipeline from web content fetching to structured data storage.

Key Features:
- Asynchronous web scraping with AsyncChromiumLoader
- LLM-powered extraction of article metadata
- Vector embedding and storage in Chroma DB
- Modular design with clear separation of concerns
"""

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_extraction_chain
from langchain_chroma import Chroma
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional
import logging
import os
import asyncio
import shutil
from datetime import datetime

# Define the schema for LLM extraction
EXTRACTION_SCHEMA = {
    "title": "NewsArticleExtraction",
    "description": "Extract structured information from news articles",
    "type": "object",
    "properties": {
        "news_article_title": {
            "type": "string",
            "description": "The title or headline of the news article"
        },
        "news_article_summary": {
            "type": "string",
            "description": "A brief summary or description of the article content"
        },
        "author": {
            "type": "string",
            "description": "The author of the article if available"
        },
        "publication_date": {
            "type": "string",
            "description": "The date when the article was published if available"
        },
        "category": {
            "type": "string",
            "description": "The category or section of the news article"
        }
    },
    "required": ["news_article_title", "news_article_summary"]
}

# 1. Configuration and Setup
def configure_logging():
    """Configure logging for the scraper"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def reset_vector_store(persist_directory="./chroma_db"):
    """Reset or initialize the vector store directory."""
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Vector store at '{persist_directory}' has been reset.")
    else:
        print(f"No existing vector store found at '{persist_directory}'. Creating new.")
    os.makedirs(persist_directory, exist_ok=True)

# 2. Web Scraping Functions
async def fetch_content(urls, logger):
    """Fetch content from specified URLs using AsyncChromiumLoader"""
    try:
        logger.info(f"Starting content fetch from {len(urls)} URLs")
        loader = AsyncChromiumLoader(urls)
        html = await loader.aload()
        logger.info("Successfully fetched HTML content")
        return html
    except Exception as e:
        logger.error(f"Error fetching content: {str(e)}")
        return []

async def transform_content(html, logger):
    """Transform HTML content using BeautifulSoupTransformer"""
    try:
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(
            html,
            tags_to_extract=["p", "h1", "h2", "h3", "article", "span", "a", "section"]
        )
        logger.info("Successfully transformed HTML content")
        return docs_transformed
    except Exception as e:
        logger.error(f"Error transforming content: {str(e)}")
        return []

# 3. Content Processing and Extraction
async def extract_content(documents, llm, logger):
    """Use LLM to extract structured information from the transformed content"""
    try:
        extracted_content = []
        
        for doc in documents:
            try:
                # For GPT-4, use function calling approach
                response = await asyncio.to_thread(
                    lambda content: llm.invoke(
                        f"Extract the news article information from the following text: {content}",
                        functions=[{
                            "name": "extract_news_article",
                            "description": "Extract structured information from news articles",
                            "parameters": EXTRACTION_SCHEMA
                        }],
                        function_call={"name": "extract_news_article"}
                    ),
                    doc.page_content
                )
                
                # Extract the function call arguments
                if hasattr(response, 'additional_kwargs') and 'function_call' in response.additional_kwargs:
                    import json
                    function_args = json.loads(response.additional_kwargs['function_call']['arguments'])
                    extracted_content.append(function_args)
            except Exception as inner_e:
                logger.warning(f"Error extracting content from document: {str(inner_e)}")
                continue
        
        logger.info(f"Successfully extracted {len(extracted_content)} articles")
        return extracted_content
    except Exception as e:
        logger.error(f"Error extracting content: {str(e)}")
        return []

def process_and_split_content(documents, logger):
    """Process and split the transformed content into manageable chunks"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split content into {len(split_docs)} chunks")
        return split_docs
    except Exception as e:
        logger.error(f"Error processing content: {str(e)}")
        return []

# 4. Storage Functions
def create_documents_from_extractions(extracted_content, logger):
    """Convert extracted content to Document objects for vectorization"""
    try:
        documents = []
        for item in extracted_content:
            # Create content by combining title and summary
            content = f"Title: {item.get('news_article_title', '')}\n\n{item.get('news_article_summary', '')}"
            
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "title": item.get('news_article_title', 'N/A'),
                    "author": item.get('author', 'N/A'),
                    "date": item.get('publication_date', 'N/A'),
                    "category": item.get('category', 'N/A'),
                    "source": "The Guardian",
                    "scraped_at": datetime.now().isoformat()
                }
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} document objects")
        return documents
    except Exception as e:
        logger.error(f"Error creating documents: {str(e)}")
        return []

def store_in_vector_db(documents, persist_directory="./chroma_db", logger=None):
    """Store documents in Chroma vector database"""
    if logger:
        logger.info(f"Storing {len(documents)} documents in vector database")
    
    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create and persist Chroma vector store
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        if logger:
            logger.info("Successfully stored documents in vector database")
        return db
    except Exception as e:
        if logger:
            logger.error(f"Error storing in vector database: {str(e)}")
        return None

# 5. Main Execution Function
async def run_scraper(urls=None, reset_db=False, persist_directory="./chroma_db"):
    """Run the complete scraping, processing, and storage pipeline"""
    # Set default URLs if none provided
    if urls is None:
        urls = [
            "https://www.theguardian.com/us",
            "https://www.theguardian.com/us/technology",
            "https://www.theguardian.com/world"
        ]
    
    # Configure logging
    logger = configure_logging()
    
    # Initialize LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    # Reset vector store if requested
    if reset_db:
        reset_vector_store(persist_directory)
    
    # Execute pipeline
    html = await fetch_content(urls, logger)
    if not html:
        return None
    
    transformed_docs = await transform_content(html, logger)
    if not transformed_docs:
        return None
    
    extracted_content = await extract_content(transformed_docs, llm, logger)
    if not extracted_content:
        return None
    
    documents = create_documents_from_extractions(extracted_content, logger)
    if not documents:
        return None
    
    # Store in vector database
    vector_db = store_in_vector_db(documents, persist_directory, logger)
    
    return vector_db

if __name__ == "__main__":
    asyncio.run(run_scraper(reset_db=True))