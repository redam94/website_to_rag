from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import os

from pydantic import Field

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from typing import List

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

from website_to_rag.utils.better_webpage_extractor import (
    ContentFilterType,
    webpage_extractor,
)

__all__ = ["create_search_agent", "SearchResults", "NoResults", "SearchDeps"]

load_dotenv()

logfire.configure(send_to_logfire="if-token-present")


@dataclass
class SearchResults:
    query: str = Field(..., description="The query that was searched")
    urls: List[str] = Field(
        ..., description="List of URLs returned by the search engine", max_length=5
    )
    summary_of_results: str = Field(
        ...,
        description="Thorough summary of the returned urls using markdown syntax.",
        max_length=1500,
    )


@dataclass
class NoResults:
    query: str = Field(..., description="The query that was searched")
    message: str = Field(..., description="Message returned by the search engine")


@dataclass
class WebScrapperSettings:
    filter_type: ContentFilterType = Field(
        default=ContentFilterType.HYBRID,
        description="Type of content filter to apply to the web page content",
    )
    custom_filter_provider: str | None = Field(
        default=None,
        description="Custom filter provider to use for content filtering",
    )
    custom_filter_instruction: str | None = Field(
        default=None,
        description="Custom instructions to use for content filtering",
    )
    bypass_cache: bool = Field(
        default=False,
        description="Bypass cache for web page content",
    )


@dataclass
class SearchDeps:
    ddgs: DDGS
    webscrapper_settings: WebScrapperSettings = Field(
        default_factory=lambda: WebScrapperSettings(
            filter_type=ContentFilterType.HYBRID,
            custom_filter_provider=None,
            custom_filter_instruction=None,
            bypass_cache=False,
        )
    )


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
            "You must use `scrape_webpage` to scrape the content of the URLs, if no content is found, return `NoResults`."
            "Use the content of the URLs to generate a detailed summary of the search results that best answers the query use mardown to format the summary. "
            "You must return the query, the list of URLs, and a summary of the search results in the `SearchResults` object. "
        ),
    )

    @search_agent.tool
    def search_tool(
        context: RunContext[SearchDeps], query: str
    ) -> List[dict[str, str]] | NoResults:
        """
        Returns at most 10 search results for the given query.
        Search results are returned as a list of dictionaries,
        where each dictionary contains the title of the webpage, url, and body of the search result."""

        ddgs = context.deps.ddgs
        try:
            results = ddgs.text(query, region="en-us", max_results=10)
        except DuckDuckGoSearchException as e:
            return NoResults(query=query, message=str(e))

        return results

    @search_agent.tool
    async def scrape_webpage(context: RunContext[SearchDeps], url: str) -> str:
        """
        Scrapes the given URL and returns a WebPage object containing the content.
        The html content is filtered and processed to extract relevant information.
        """
        scraper_settings = context.deps.webscrapper_settings

        webpage = await webpage_extractor(
            url,
            filter_type=scraper_settings.filter_type,
            custom_filter_provider=scraper_settings.custom_filter_provider,
            custom_filter_instruction=scraper_settings.custom_filter_instruction,
            bypass_cache=scraper_settings.bypass_cache,
        )

        return webpage.original_md

    return search_agent


if __name__ == "__main__":
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from rich.console import Console
    from rich.pretty import pprint
    from rich.prompt import Prompt
    from rich.markdown import Markdown

    from website_to_rag.utils.better_webpage_extractor import webpage_extractor

    search_deps = SearchDeps(
        ddgs=DDGS(),
        webscrapper_settings=WebScrapperSettings(
            filter_type=ContentFilterType.STRICT_RULES,
            custom_filter_provider=None,
            custom_filter_instruction=None,
            bypass_cache=False,
        ),
    )

    async def main():
        console = Console()
        selected_model = Prompt.ask(
            "Select a model to use",
            choices=["openai", "local", "google"],
            default="google",
        )
        match selected_model:
            case "openai":
                model = OpenAIModel(model_name="gpt-4o")

            case "local":
                model = OpenAIModel(
                    model_name="qwen3:4b",
                    provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
                )
            case "google":
                model = "google-gla:gemini-2.5-flash-preview-05-20"
            case _:
                raise ValueError(f"Invalid model selected: {selected_model}")

        # model = OpenAIModel(
        #     model_name="qwen3:4b", provider=OpenAIProvider(base_url='http://localhost:11434/v1')
        # )
        # model = OpenAIModel(model_name="gpt-4.1-mini")
        # model = "google-gla:gemini-2.5-flash-preview-05-20"
        search_agent = create_search_agent(model)
        prompt = None
        retries = 0
        while True:
            if prompt is None or retries >= 3:
                prompt = Prompt.ask("What would you like to know?")
                retries = 0

            if prompt.lower() == "exit":
                break
            try:
                retries += 1
                results = await search_agent.run(
                    prompt,
                    deps=search_deps,
                )

            except UnexpectedModelBehavior as e:
                console.print(e)

                continue

            if isinstance(results.output, NoResults):
                console.print(results.output.message)
                continue

            pprint(results.output.query)
            pprint(results.output.urls)
            console.print(Markdown(results.output.summary_of_results))
            prompt = None
            # console.print(Markdown(results.data.summary_of_results))
            # tasks = asyncio.gather(*[webpage_extractor(url) for url in results.data.urls])

    asyncio.run(main())
