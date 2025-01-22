from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import os

from pydantic import Field

from pydantic_ai import Agent, RunContext
from typing import List

from duckduckgo_search import DDGS
from openai import AsyncOpenAI

from website_to_rag.utils.webpage_extractor import WebPage

__all__ = ["search_agent", "SearchResults", "NoResults", "SearchDeps"]

load_dotenv()

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class SearchResults:
    query: str
    urls: List[str] = Field(..., 
        description="List of URLs returned by the search engine", 
        max_length=5)
        

@dataclass
class NoResults:
    query: str
    message: str = Field(..., 
        description="Message returned by the search engine")
        

@dataclass
class SearchDeps:
    ddgs: DDGS
    extracted_pages: list[WebPage] = Field(default_factory=list)
    openai: AsyncOpenAI = Field(default_factory=AsyncOpenAI)

model = os.environ.get('MODEL', 'openai:gpt-4o-mini')

search_agent = Agent[SearchDeps, SearchResults|NoResults](
    model,
    name="search_agent",
    result_type=SearchResults,
    deps_type=SearchDeps,
    system_prompt=(
        "You are an expert search engine that takes a user query and returns the top 5 URLs that are most relevant to the query. "
        "You may edit the query before running the search in order to improve the search results. "
        "You can break up the prompt into multiple seperate search queries if needed, make sure to run `search_tool` for each query. "
        #"Use the `extract_webpage` tool to extract the content of each URL. "
        #"Then you must use the `get_most_relevant_sections` tool to extract the most relevant sections of each URL after running the `extract_webpage` tool. "
        #"You must use the `get_most_relevant_sections` tool to answer the users query! "
        "You must use `search_tool` to search for the URLs, if no urls are found, return `NoResults`."
    )
)

@search_agent.tool
def search_tool(context: RunContext[SearchDeps], query: str) -> List[dict[str, str]]:
    """
    Returns at most 10 search results for the given query. 
    Search results are returned as a list of dictionaries, 
    where each dictionary contains the title of the webpage, url, and body of the search result."""

    ddgs = context.deps.ddgs
    results = ddgs.text(query, region='en-us', max_results=10)

    return results
    
# @search_agent.tool
# async def extract_webpage(context: RunContext[SearchDeps], url: str) -> WebPage:
#     """
#     Extracts the content of a webpage given a URL. 
#     Updateds the `extracted_pages` attribute of the context with the extracted webpage."""
#     context.deps.extracted_pages.append(await webpage_extractor(url))

# @search_agent.tool
# async def get_most_relevant_sections(context: RunContext[SearchDeps], query: str) -> List[str]:
#     """
#     Given a search query retrive the most relevent context from aquired webpages. 
#     Returns the most relevant sections of the webpage as a list of strings.
#     This tool should be used after the `extract_webpage` tool has been run and can be used to answer the user query."""
#     query_embedding = await context.deps.openai.embeddings.create(input=query, model='text-embedding-3-large')
#     def _distance_list(page):
#         my_list = [(section.compute_distance(query_embedding.data[0].embedding), section) for section in page.sections]

#         return [section for _, section in sorted(my_list, key=lambda x: x[0], reverse=True)][:3]
#     return [section.markdown for page in context.deps.extracted_pages for section in _distance_list(page)]

if __name__ == "__main__":
    from rich.console import Console
    from rich.pretty import pprint
    from rich.prompt import Prompt
    async def main():
        console = Console()
        while True:
            prompt = Prompt.ask("What would you like to know?")
            if prompt.lower() == "exit":
                break
        
            results = await search_agent.run(prompt, deps=SearchDeps(ddgs=DDGS(), extracted_pages=[], openai=AsyncOpenAI()))
            if isinstance(results.data, NoResults):
                console.print(results.data.message)
                continue
            pprint(results.data.query)
            pprint(results.data.urls)
            
            #console.print(Markdown(results.data.summary_of_results))
            #tasks = asyncio.gather(*[webpage_extractor(url) for url in results.data.urls])
            
            
                
            
    asyncio.run(main())
