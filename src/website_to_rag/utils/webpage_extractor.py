from __future__ import annotations as _annotations

from dataclasses import dataclass

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
)
from crawl4ai.content_filter_strategy import LLMContentFilter

import httpx
from pydantic import Field
from openai import OpenAI, AsyncOpenAI
import openai
import numpy as np
import ollama

from typing import Protocol, Literal
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser

__all__ = [
    "webpage_extractor",
    "WebPage",
    "Section",
    "embed_webpage",
    "async_embed_webpage",
    "OpenAIEmbedder",
    "AsyncOpenAIEmbedder",
    "Embedder",
    "AsyncEmbedder",
]

BROWSER_CONFIG = BrowserConfig(verbose=True)

LLM_FILTER = LLMContentFilter(
    provider="ollama/llama3.2",
    instruction="""
    Extract the main educational content while preserving its original wording and substance completely.
    1. Maintain the exact language and terminology
    2. Keep all technical explanations and examples intact
    3. Preserve the original flow and structure
    4. Remove only clearly irrelevant elements like navigation menus and ads
    """,
    chunk_token_threshold=4096,
)

RUN_CONFIG = CrawlerRunConfig(
    excluded_tags=["nav", "footer", "header"],
    exclude_external_links=True,
    markdown_generator=DefaultMarkdownGenerator(
        content_filter=LLM_FILTER,
    ),
    check_robots_txt=True,
)

# RUN_CONFIG = CrawlerRunConfig(
#    excluded_tags=["form", "header", "footer"], keep_data_attributes=False
# )


class Embedder(Protocol):
    def __call__(self, section: Section) -> list[float]: ...


class AsyncEmbedder(Protocol):
    async def __call__(self, section: Section) -> list[float]: ...


class OpenAIEmbedder:
    def __init__(
        self,
        model: Literal["text-embedding-3-small", "text-embedding-3-large"],
        section_chunker=None,
    ):
        self.model = model
        self.client = OpenAI()
        self.section_chunker = section_chunker

    def __call__(self, section: Section) -> list[list[float]]:
        if self.section_chunker is None:
            return [
                openai.embeddings.create(input=section.markdown, model=self.model)
                .data[0]
                .embedding
            ]

        doc = Document(text=section.markdown)
        nodes = self.section_chunker.get_nodes_from_documents([doc])
        subsections = [node.get_text() for node in nodes]
        embeddings = [
            openai.embeddings.create(input=subsection, model=self.model)
            .data[0]
            .embedding
            for subsection in subsections
        ]
        return embeddings


class OllamaEmbedder:
    def __init__(
        self,
        model: Literal["nomic-embed-text"],
        section_chunker=None,
    ):
        self.model = model
        self.section_chunker = section_chunker

    def __call__(self, section: Section) -> list[list[float]]:
        if self.section_chunker is None:
            return [
                ollama.embeddings(prompt=section.markdown, model=self.model).embedding
            ]

        doc = Document(text=section.markdown)
        nodes = self.section_chunker.get_nodes_from_documents([doc])
        subsections = [node.get_text() for node in nodes]
        embeddings = [
            ollama.embeddings(prompt=subsection, model=self.model).embedding
            for subsection in subsections
        ]
        return embeddings


class AsyncOpenAIEmbedder:
    def __init__(
        self,
        model: Literal["text-embedding-3-small", "text-embedding-3-large"],
        section_chunker=None,
    ):
        self.model = model
        self.client = AsyncOpenAI()
        self.section_chunker = section_chunker

    async def __call__(self, section: Section) -> list[list[float]]:
        if self.section_chunker is None:
            return [
                (
                    await self.client.embeddings.create(
                        input=section.markdown, model=self.model
                    )
                )
                .data[0]
                .embedding
            ]

        doc = Document(text=section.markdown)
        nodes = self.section_chunker.get_nodes_from_documents([doc])
        subsections = [node.get_text() for node in nodes]
        embeddings = [
            (await self.client.embeddings.create(input=subsection, model=self.model))
            .data[0]
            .embedding
            for subsection in subsections
        ]
        return embeddings


def _cosine_similarity(a, b):
    if len(a) == 0 or len(b) == 0:
        return -np.inf
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def _embed_section(section: Section, embedder: Embedder) -> list[float]:
    return embedder(section)


async def _async_embed_section(
    section: Section, embedder: AsyncEmbedder
) -> list[float]:
    return await embedder(section)


def embed_webpage(
    webpage: WebPage, embedder: Embedder
) -> list[tuple[Section, list[list[float]]]]:
    return [(section, embedder(section)) for section in webpage.sections]


async def async_embed_webpage(
    webpage: WebPage, embedder: AsyncEmbedder
) -> list[tuple[Section, list[list[float]]]]:
    result = []
    for section in webpage.sections:
        try:
            result.append((section, await embedder(section)))
        except openai.error.InvalidRequestError:
            continue
    return [(section, await embedder(section)) for section in webpage.sections]


@dataclass
class Section:
    title: str = Field(..., description="Title of the section")
    content: str = Field(..., description="Content of the section")

    @property
    def markdown(self) -> str:
        return f"{self.title}\n\n{self.content}"

    @property
    def level(self) -> int:
        return self.title.count("#")

    def __hash__(self):
        return self.markdown.__hash__()


def compute_distance(
    section_embedding: list[list[float]], query_embedding: list[float], power=1
) -> float:
    return (
        sum(
            _cosine_similarity(emb, query_embedding) ** power
            for emb in section_embedding
        )
        / len(section_embedding)
    ) ** (1, 0 / power)


@dataclass
class WebPage:
    url: str = Field(..., description="URL of the webpage")
    body: dict[Section, list[Section]] = Field(
        ..., description="Adjacency list of the webpage"
    )
    original_md: str = Field(default="", description="Original markdown of the webpage")

    @property
    def sections(self) -> list[Section]:
        return {section for sections in self.body.values() for section in sections} | {
            section for section in self.body.keys()
        }


def _edgelist_from_sections(sections: list[Section]) -> list[tuple[Section, Section]]:
    """Create an edge list from a webpage using the section levels
    Example:
       '''# Title
       ## Subtitle
       # Another title
       ## Another subtitle
       ### A subsubtitle
       ## A Second subtitle
       '''
       -> [('Title', 'Subtitle'), ('Another title', 'Another subtitle'), ('Another subtitle', 'A subsubtitle'), ('Another subtitle', 'A Second subtitle')]
    """
    if not sections:
        return []
    memory = [sections[0]]
    result = []
    for section in sections[1:]:
        if section.level > memory[-1].level:
            memory.append(section)
        else:
            memory = memory[: section.level]
            memory[-1] = section
        try:
            result.append((memory[-2], memory[-1]))
        except IndexError:
            result.append((Section(title="", content=""), memory[-1]))
    return result


def _edgelist_to_adjacency(
    edglist: list[tuple[Section, Section]],
) -> dict[Section, list[Section]]:
    result = {}
    for edge in edglist:
        if edge[0] not in result:
            result[edge[0]] = []
        result[edge[0]].append(edge[1])
    return result


def _markdown_to_webpage(url: str, markdown: str) -> WebPage:
    parser = MarkdownNodeParser()

    # content = re.sub(r"[\-\*].*\(.*\)\n\s*", "", markdown)
    # content = re.sub(r"\[\]\(.*\)", "", content)
    content = markdown
    doc = Document(text=content)
    nodes = parser.get_nodes_from_documents([doc])
    sections = [
        Section(
            title=node.get_text().split("\n")[0],
            content="\n".join(node.get_text().split("\n")[1:]),
        )
        for node in nodes
    ]
    edgelist = _edgelist_from_sections(sections)
    adjacency = _edgelist_to_adjacency(edgelist)
    return WebPage(url=url, body=adjacency, original_md=markdown)


async def webpage_extractor(url: str) -> WebPage | str:
    try:
        async with AsyncWebCrawler(config=BROWSER_CONFIG) as crawler:
            result = await crawler.arun(url=url, config=RUN_CONFIG)
            return _markdown_to_webpage(url, result.markdown)
    except httpx.HTTPError as e:
        return f"Error: {e} for url: {url} this may be a pdf file or the url might be invalid"


if __name__ == "__main__":
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from llama_index.core.node_parser import SentenceSplitter
    import numpy as np
    import asyncio

    openai = OpenAI()

    async def main():
        console = Console()
        while True:
            url = Prompt.ask("URL:")
            if url.lower() == "exit":
                break

            results = await webpage_extractor(url)
            if isinstance(results, str):
                console.print(results)
                continue

            console.print(Markdown(results.original_md))
            console.print(Markdown(f"# Extracted content from {url}\n---"))
            for section in results.sections:
                console.print(Markdown("## New Section\n\n"))
                md = Markdown(section.markdown)
                console.print(md)
                console.print(Markdown("---"))

            embeded_webpage = embed_webpage(
                results,
                OllamaEmbedder(
                    "nomic-embed-text",
                    section_chunker=SentenceSplitter(chunk_size=2048, chunk_overlap=40),
                ),
            )
            # for section, embedding in embeded_webpage:
            #     console.print(Markdown(section.markdown))
            #     console.print(
            #        embedding
            #     )

            while True:
                query = Prompt.ask("What would you like to know?")
                if query.lower() == "exit":
                    break
                query_embedding = (
                    ollama.embeddings(prompt=query, model="nomic-embed-text")
                ).embedding

                console.print(Markdown("# Average Embedding Distance"))

                sorted_sections = sorted(
                    embeded_webpage,
                    key=lambda x: max(
                        _cosine_similarity(emb, query_embedding) for emb in x[1]
                    ),
                    # / len(x[1]),
                    reverse=True,
                )
                for section, embedding in sorted_sections[:3]:
                    console.print(
                        Markdown(
                            f"# Distance: {sum(_cosine_similarity(emb, query_embedding) for emb in embedding)/len(embedding): .4f}\n\n## Length: {len(embedding)}"
                        )
                    )
                    console.print(Markdown(section.markdown))
                    console.print(Markdown("---"))

                console.print(Markdown("# Distance from Average Embedding"))
                sorted_sections = sorted(
                    embeded_webpage,
                    key=lambda x: _cosine_similarity(
                        np.mean(x[1], axis=0), query_embedding
                    ),
                    reverse=True,
                )
                for section, embedding in sorted_sections[:3]:
                    console.print(
                        Markdown(
                            f"# Distance: {_cosine_similarity(np.mean(embedding, axis=0), query_embedding): .4f}\n\n## Length: {len(embedding)}"
                        )
                    )
                    console.print(Markdown(section.markdown))
                    console.print(Markdown("---"))

    asyncio.run(main())
