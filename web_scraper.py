"""
News Scraper: A system for extracting structured data from The Guardian website

This program scrapes content from The Guardian, processes it using BeautifulSoup,
and stores the results in a Chroma vector database for later retrieval.
It handles the entire pipeline from web content fetching to structured data storage.

Key Features:
- Asynchronous web scraping with AsyncChromiumLoader
- Direct HTML parsing with BeautifulSoup
- Vector embedding and storage in Chroma DB
- Modular design with clear separation of concerns
"""

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging
import os
import asyncio
import shutil
import re
from datetime import datetime

# Set user agent for web requests
os.environ["USER_AGENT"] = "Guardian News Scraper Bot (educational purposes)"

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

# 3. Content Processing and Extraction
def extract_articles_from_html(html_docs, logger):
    """Extract article information directly from HTML using BeautifulSoup"""
    try:
        articles = []
        
        for html_doc in html_docs:
            soup = BeautifulSoup(html_doc.page_content, 'html.parser')
            
            # The Guardian's article elements typically have specific classes
            # Find all article elements
            article_elements = soup.find_all('div', class_=lambda c: c and ('fc-item' in c or 'dcr-' in c or 'js-headline-text' in c))
            
            if not article_elements:
                # Try alternative selectors
                article_elements = soup.find_all(['article', 'div'], class_=lambda c: c and ('article' in c.lower() if c else False))
            
            # Process each article element
            for article_element in article_elements:
                # Extract title
                title_element = article_element.find(['h1', 'h2', 'h3'], class_=lambda c: c and ('headline' in c.lower() if c else False))
                if not title_element:
                    title_element = article_element.find(['h1', 'h2', 'h3'])
                
                title = title_element.get_text().strip() if title_element else None
                
                # Skip if no title found
                if not title:
                    continue
                
                # Extract summary
                summary_element = article_element.find('div', class_=lambda c: c and ('standfirst' in c.lower() if c else False))
                if not summary_element:
                    # Try to find paragraphs
                    p_elements = article_element.find_all('p')
                    summary = ' '.join([p.get_text().strip() for p in p_elements[:2]]) if p_elements else ''
                else:
                    summary = summary_element.get_text().strip()
                
                # Extract date (if available)
                date_element = article_element.find(['time', 'span'], attrs={'datetime': True})
                date = date_element.get('datetime') if date_element else None
                if not date:
                    date_element = article_element.find(text=re.compile(r'\d{1,2}\s+[A-Za-z]{3}\s+\d{4}'))
                    date = date_element if date_element else None
                
                # Extract author (if available)
                author_element = article_element.find(['span', 'div', 'a'], class_=lambda c: c and ('contributor' in c.lower() or 'byline' in c.lower() if c else False))
                author = author_element.get_text().strip() if author_element else None
                
                # Extract category
                category_element = article_element.find(['a', 'span'], class_=lambda c: c and ('kicker' in c.lower() or 'section' in c.lower() if c else False))
                category = category_element.get_text().strip() if category_element else None
                
                # Create article dict
                article = {
                    'news_article_title': title,
                    'news_article_summary': summary if summary else "No summary available",
                    'author': author if author else "Unknown",
                    'publication_date': date if date else "Unknown",
                    'category': category if category else "Uncategorized",
                    'source_url': html_doc.metadata.get('source', '')
                }
                
                articles.append(article)
        
        # Remove duplicates based on title
        unique_articles = []
        seen_titles = set()
        for article in articles:
            if article['news_article_title'] not in seen_titles:
                seen_titles.add(article['news_article_title'])
                unique_articles.append(article)
        
        logger.info(f"Successfully extracted {len(unique_articles)} unique articles")
        return unique_articles
    except Exception as e:
        logger.error(f"Error extracting articles: {str(e)}")
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
                    "source_url": item.get('source_url', ''),
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
            "https://www.theguardian.com/world",
            "https://www.theguardian.com/business"
        ]
    
    # Configure logging
    logger = configure_logging()
    
    # Reset vector store if requested
    if reset_db:
        reset_vector_store(persist_directory)
    
    # Execute pipeline
    html = await fetch_content(urls, logger)
    if not html:
        return None
    
    # Extract articles from HTML
    extracted_content = extract_articles_from_html(html, logger)
    if not extracted_content:
        return None
    
    # Create document objects
    documents = create_documents_from_extractions(extracted_content, logger)
    if not documents:
        return None
    
    # Store in vector database
    vector_db = store_in_vector_db(documents, persist_directory, logger)
    
    return vector_db

if __name__ == "__main__":
    asyncio.run(run_scraper(reset_db=True))