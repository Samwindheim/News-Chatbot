#!/usr/bin/env python3
"""
News Application Orchestrator

This script provides a command-line interface to control the news scraping and chat
functionality. It serves as the main entry point for the application, allowing users
to scrape new data, start the chat interface, or perform both actions in sequence.

Usage:
  python news_app.py scrape [--reset] [--sources SOURCE1 SOURCE2 ...]
  python news_app.py chat [--fresh]
  python news_app.py both [--reset] [--sources SOURCE1 SOURCE2 ...]

Options:
  --reset            Reset the database before scraping
  --sources          Specify which news sources to scrape (defaults to The Guardian sections)
  --fresh            Start chat with freshly scraped data
  --db-dir           Specify the database directory (default: ./chroma_db)
  --verbose          Enable verbose logging
"""

import argparse
import asyncio
import sys
import logging
import os
from datetime import datetime
from web_scraper import run_scraper
from new_agent import NewsChatbot
from dotenv import load_dotenv
load_dotenv()

# Default Guardian sections to scrape
DEFAULT_SOURCES = [
    "https://www.theguardian.com/us",
    "https://www.theguardian.com/us/technology",
    "https://www.theguardian.com/world",
    "https://www.theguardian.com/business"
]

def setup_logging(verbose=False):
    """Configure logging for the application"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('news_app')

async def run_scraping(args, logger):
    """Run the web scraper with provided arguments"""
    logger.info("Starting web scraping process...")
    
    # Use custom sources if provided, otherwise use defaults
    sources = args.sources if args.sources else DEFAULT_SOURCES
    logger.info(f"Scraping the following sources: {', '.join(sources)}")
    
    try:
        # Run the scraper with provided parameters
        start_time = datetime.now()
        db = await run_scraper(
            urls=sources,
            reset_db=args.reset,
            persist_directory=args.db_dir
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        if db:
            logger.info(f"Scraping completed successfully in {elapsed:.2f} seconds")
            return True
        else:
            logger.error("Scraping failed - no database was returned")
            return False
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        return False

def start_chat(args, logger):
    """Start the chat interface with provided arguments"""
    logger.info("Starting chat interface...")
    
    try:
        # Initialize the chatbot
        chatbot = NewsChatbot(
            chroma_directory=args.db_dir,
            refresh_data=args.fresh
        )
        
        # Display a welcome message
        article_count = len(chatbot.chroma_db.get()["ids"]) if hasattr(chatbot, 'chroma_db') else "unknown"
        print("\n" + "="*50)
        print(f"News Chatbot - {article_count} articles available")
        print("Ask questions about recent news or type 'exit' to quit")
        print("="*50 + "\n")
        
        # Start the interactive chat session
        chatbot.interactive_chat()
        return True
    except Exception as e:
        logger.error(f"Error starting chat: {str(e)}")
        return False

async def main():
    """Main entry point for the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="News Application")
    
    # Main action argument
    parser.add_argument('action', choices=['scrape', 'chat', 'both'], 
                        help='Action to perform')
    
    # Optional arguments
    parser.add_argument('--reset', action='store_true', 
                        help='Reset the database before scraping')
    parser.add_argument('--sources', nargs='+', 
                        help='Specific news sources to scrape')
    parser.add_argument('--fresh', action='store_true',
                        help='Start chat with freshly scraped data')
    parser.add_argument('--db-dir', default='./chroma_db',
                        help='Database directory path')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    
    # Make sure database directory exists
    os.makedirs(args.db_dir, exist_ok=True)
    
    # Perform the requested action(s)
    success = True
    
    if args.action in ['scrape', 'both']:
        success = await run_scraping(args, logger)
        
    if args.action in ['chat', 'both'] and success:
        # Don't need to refresh data if we just scraped it
        if args.action == 'both':
            args.fresh = False
        
        success = start_chat(args, logger)
    
    return 0 if success else 1

if __name__ == "__main__":
    # Run the async main function
    sys.exit(asyncio.run(main())) 