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

__all__ = ["create_search_agent", "SearchResults", "NoResults", "SearchDeps"]

load_dotenv()

logfire.configure(send_to_logfire="if-token-present")


@dataclass
class SearchResults:
    query: str
    urls: List[str] = Field(
        ..., description="List of URLs returned by the search engine", max_length=5
    )


@dataclass
class NoResults:
    query: str
    message: str = Field(..., description="Message returned by the search engine")


@dataclass
class SearchDeps:
    ddgs: DDGS
#    extracted_pages: list[WebPage] = Field(default_factory=list)
#    openai: AsyncOpenAI = Field(default_factory=AsyncOpenAI)


def create_search_agent(model=None):
    if model is None:
        model = os.environ.get("MODEL", "openai:gpt-4o-mini")

    search_agent = Agent[SearchDeps, SearchResults | NoResults](
        model,
        name="search_agent",
        result_type=SearchResults,
        deps_type=SearchDeps,
        system_prompt=(
            "You are an expert search engine that takes a user query and returns the top 5 URLs that are most relevant to the query. "
            "You may edit the query before running the search in order to improve the search results. "
            "You can break up the prompt into multiple seperate search queries if needed, make sure to run `search_tool` for each query. "
            "You must use `search_tool` to search for the URLs, if no urls are found, return `NoResults`."
        ),
    )

    @search_agent.tool
    def search_tool(
        context: RunContext[SearchDeps], query: str
    ) -> List[dict[str, str]]:
        """
        Returns at most 10 search results for the given query.
        Search results are returned as a list of dictionaries,
        where each dictionary contains the title of the webpage, url, and body of the search result."""

        ddgs = context.deps.ddgs
        results = ddgs.text(query, region="en-us", max_results=10)

        return results

    return search_agent


if __name__ == "__main__":
    from rich.console import Console
    from rich.pretty import pprint
    from rich.prompt import Prompt
    from pydantic_ai.models.openai import OpenAIModel

    async def main():
        console = Console()
        model = OpenAIModel(model_name='qwen2.5:7b', base_url='http://localhost:11434/v1')
        search_agent = create_search_agent(model)
        while True:
            prompt = Prompt.ask("What would you like to know?")
            if prompt.lower() == "exit":
                break

            results = await search_agent.run(
                prompt,
                deps=SearchDeps(ddgs=DDGS()),
            )
            if isinstance(results.data, NoResults):
                console.print(results.data.message)
                continue
            pprint(results.data.query)
            pprint(results.data.urls)

            # console.print(Markdown(results.data.summary_of_results))
            # tasks = asyncio.gather(*[webpage_extractor(url) for url in results.data.urls])

    asyncio.run(main())
