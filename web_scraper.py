"""
Script to scrape web data from The Guardian website
using LangChain's web scraping capabilities with AsyncChromiumLoader
"""
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import logging
import os
import asyncio
from datetime import datetime

class GuardianScraper:
    def __init__(self):
        self.base_url = "https://www.theguardian.com/us"
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def fetch_content(self) -> List[Dict]:
        """
        Fetch content from The Guardian website using AsyncChromiumLoader
        """
        try:
            self.logger.info("Starting content fetch from The Guardian")
            loader = AsyncChromiumLoader([self.base_url])
            html = await loader.aload()
            self.logger.info("Successfully fetched HTML content")
            return html
        except Exception as e:
            self.logger.error(f"Error fetching content: {str(e)}")
            return []

    def transform_content(self, html: List[Dict]) -> List[Dict]:
        """
        Transform HTML content using BeautifulSoupTransformer
        """
        try:
            bs_transformer = BeautifulSoupTransformer()
            # Extract content from specific tags that are common in news articles
            docs_transformed = bs_transformer.transform_documents(
                html,
                tags_to_extract=["p", "h1", "h2", "h3", "article", "span"]
            )
            self.logger.info("Successfully transformed HTML content")
            return docs_transformed
        except Exception as e:
            self.logger.error(f"Error transforming content: {str(e)}")
            return []

    def process_content(self, documents: List[Dict]) -> List[Dict]:
        """
        Process and split the transformed content into manageable chunks
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            split_docs = text_splitter.split_documents(documents)
            self.logger.info(f"Split content into {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}")
            return []

    def save_content(self, documents: List[Dict], output_dir: str = "scraped_data"):
        """
        Save the processed content to files
        """
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"guardian_content_{timestamp}.txt"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                for doc in documents:
                    f.write(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
                    f.write(f"Content:\n{doc.page_content}\n")
                    f.write("-" * 80 + "\n")

            self.logger.info(f"Content saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving content: {str(e)}")

async def main():
    scraper = GuardianScraper()
    
    # Fetch content
    html = await scraper.fetch_content()
    if not html:
        return
    
    # Transform HTML
    transformed_docs = scraper.transform_content(html)
    if not transformed_docs:
        return
    
    # Process content
    processed_docs = scraper.process_content(transformed_docs)
    if not processed_docs:
        return
    
    # Save content
    scraper.save_content(processed_docs)

if __name__ == "__main__":
    asyncio.run(main())