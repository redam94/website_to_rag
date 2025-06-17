from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, Optional
from abc import ABC, abstractmethod
from enum import Enum
import re
from bs4 import BeautifulSoup

import numpy as np
import httpx
from pydantic import Field
from openai import OpenAI, AsyncOpenAI
import ollama
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    LLMConfig,
)
from crawl4ai.content_filter_strategy import LLMContentFilter

# Type aliases
EmbeddingVector: TypeAlias = list[float]
EmbeddingList: TypeAlias = list[EmbeddingVector]

# Constants
EMBEDDING_MODELS = Literal[
    "text-embedding-3-small", "text-embedding-3-large", "nomic-embed-text"
]


class BaseContentFilter(ABC):
    """Abstract base class for all content filters."""

    @abstractmethod
    def filter_content(self, content: str) -> str:
        """Filter the given content and return the filtered result."""
        pass


class RuleBasedFilter(BaseContentFilter):
    """Rule-based content filter using regex patterns and BeautifulSoup."""

    def __init__(
        self,
        remove_patterns: list[str] = None,
        keep_patterns: list[str] = None,
        remove_tags: list[str] = None,
        remove_classes: list[str] = None,
        remove_ids: list[str] = None,
    ):
        self.remove_patterns = [re.compile(p) for p in (remove_patterns or [])]
        self.keep_patterns = [re.compile(p) for p in (keep_patterns or [])]
        self.remove_tags = remove_tags or []
        self.remove_classes = remove_classes or []
        self.remove_ids = remove_ids or []

    def filter_content(self, content: str) -> str:
        # First use BeautifulSoup to handle HTML content
        if "<" in content and ">" in content:
            soup = BeautifulSoup(content, "html.parser")

            # Remove unwanted tags
            for tag in self.remove_tags:
                for element in soup.find_all(tag):
                    element.decompose()

            # Remove elements with specific classes
            for class_name in self.remove_classes:
                for element in soup.find_all(class_=class_name):
                    element.decompose()

            # Remove elements with specific IDs
            for id_name in self.remove_ids:
                for element in soup.find_all(id=id_name):
                    element.decompose()

            content = str(soup)

        # Apply regex patterns
        if self.keep_patterns:
            # Keep only content that matches any of the keep patterns
            kept_content = []
            for pattern in self.keep_patterns:
                kept_content.extend(pattern.findall(content))
            content = "\n".join(kept_content)

        # Remove unwanted patterns
        for pattern in self.remove_patterns:
            content = pattern.sub("", content)

        return content.strip()


class LLMFilter(BaseContentFilter):
    """Wrapper for LLM-based content filtering."""

    def __init__(
        self, provider: str, instruction: str, chunk_token_threshold: int = 4096
    ):
        self.llm_filter = LLMContentFilter(
            llm_config=LLMConfig(provider=provider),
            instruction=instruction,
            chunk_token_threshold=chunk_token_threshold,
            # verbose=True
        )

    def filter_content(self, content: str) -> str:
        # print("Filtering content with LLM")
        return self.llm_filter.filter_content(content)


class CompositeFilter(BaseContentFilter):
    """Combines multiple filters that are applied in sequence."""

    def __init__(self, filters: list[BaseContentFilter]):
        self.filters = filters

    def filter_content(self, content: str) -> str:
        for filter_instance in self.filters:
            content = filter_instance.filter_content(content)
        return content


class ContentFilterType(Enum):
    """Enumeration of available content filter types."""

    EDUCATIONAL_LLM = "educational_llm"
    TECHNICAL_LLM = "technical_llm"
    MINIMAL_RULES = "minimal_rules"
    STRICT_RULES = "strict_rules"
    HYBRID = "hybrid"


class ContentFilterFactory:
    """Factory for creating content filters."""

    @staticmethod
    def create_filter(
        filter_type: ContentFilterType | str,
        custom_instruction: Optional[str] = None,
        custom_provider: Optional[str] = None,
    ) -> BaseContentFilter:
        """Create a content filter based on the specified type or custom configuration."""

        if isinstance(filter_type, str):
            filter_type = ContentFilterType(filter_type.lower())

        base_provider = custom_provider or "ollama/llama3.2"

        # Common rule-based patterns and configurations
        common_remove_patterns = [
            r"Cookie Policy.*?(?=\n\n)",
            r"Privacy Policy.*?(?=\n\n)",
            r"Share this:.*?(?=\n\n)",
            r"\d+ shares",
            r"Related Posts:.*?(?=\n\n)",
        ]

        common_remove_tags = ["nav", "footer", "header", "script", "style", "iframe"]
        common_remove_classes = [
            "advertisement",
            "social-share",
            "cookie-notice",
            "sidebar",
        ]

        filter_configs = {
            ContentFilterType.EDUCATIONAL_LLM: LLMFilter(
                provider=base_provider,
                instruction="""
                Extract the main educational content while preserving its original wording and substance completely.
                1. Maintain the exact language and terminology
                2. Keep all technical explanations and examples intact
                3. Preserve the original flow and structure
                4. Remove only clearly irrelevant elements like navigation menus and ads
                """,
            ),
            ContentFilterType.TECHNICAL_LLM: LLMFilter(
                provider=base_provider,
                instruction="""
                Focus on technical content extraction with emphasis on code blocks and technical details.
                1. Preserve all code snippets, commands, and technical specifications
                2. Keep technical diagrams and references
                3. Maintain API documentation and examples
                4. Remove marketing content and non-technical discussions
                5. Remove any irrelevant links and navigation elements
                """,
            ),
            ContentFilterType.MINIMAL_RULES: RuleBasedFilter(
                remove_patterns=common_remove_patterns,
                remove_tags=["script", "style", "iframe"],
                remove_classes=["advertisement"],
                remove_ids=["cookie-banner", "newsletter-signup"],
            ),
            ContentFilterType.STRICT_RULES: RuleBasedFilter(
                remove_patterns=common_remove_patterns
                + [
                    r"Comments.*?(?=\n\n)",
                    r"Author:.*?(?=\n\n)",
                    r"Posted on.*?(?=\n\n)",
                ],
                remove_tags=common_remove_tags + ["aside", "comments", "meta"],
                remove_classes=common_remove_classes
                + [
                    "comments",
                    "meta",
                    "tags",
                    "author-bio",
                    "related-posts",
                    "newsletter",
                    "promotion",
                ],
                remove_ids=["comments", "sidebar", "related-posts"],
            ),
            ContentFilterType.HYBRID: CompositeFilter(
                [
                    RuleBasedFilter(
                        remove_patterns=common_remove_patterns,
                        remove_tags=common_remove_tags,
                        remove_classes=common_remove_classes,
                    ),
                    LLMFilter(
                        provider=base_provider,
                        instruction="""
                    Refine the pre-filtered content while preserving important information:
                    1. Keep all educational and technical content
                    2. Maintain code examples and technical diagrams
                    3. Remove any remaining promotional or irrelevant content
                    4. Ensure content flow and readability
                    """,
                    ),
                ]
            ),
        }

        if custom_instruction:
            return LLMFilter(provider=base_provider, instruction=custom_instruction)

        return filter_configs[filter_type]


BROWSER_CONFIG = BrowserConfig(verbose=True)


@dataclass
class Section:
    """Represents a section of a webpage with title and content."""

    title: str = Field(..., description="Title of the section")
    content: str = Field(..., description="Content of the section")

    @property
    def markdown(self) -> str:
        return f"{self.title}\n\n{self.content}"

    @property
    def level(self) -> int:
        return self.title.count("#")

    def __hash__(self):
        return hash(self.markdown)


@dataclass
class WebPage:
    """Represents a webpage with its URL, content structure, and original markdown."""

    url: str = Field(..., description="URL of the webpage")
    body: dict[Section, list[Section]] = Field(
        ..., description="Adjacency list of the webpage"
    )
    original_md: str = Field(default="", description="Original markdown of the webpage")

    @property
    def sections(self) -> set[Section]:
        return {
            section for sections in self.body.values() for section in sections
        } | set(self.body.keys())


class BaseEmbedder(ABC):
    """Base class for embedding models."""

    def __init__(self, model: EMBEDDING_MODELS, section_chunker=None):
        self.model = model
        self.section_chunker = section_chunker

    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingVector:
        """Embed a single piece of text."""
        pass

    def __call__(self, section: Section) -> EmbeddingList:
        if self.section_chunker is None:
            return [self.embed_text(section.markdown)]

        doc = Document(text=section.markdown)
        nodes = self.section_chunker.get_nodes_from_documents([doc])
        return [self.embed_text(node.get_text()) for node in nodes]


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding model implementation."""

    def __init__(
        self,
        model: Literal["text-embedding-3-small", "text-embedding-3-large"],
        section_chunker=None,
    ):
        super().__init__(model, section_chunker)
        self.client = OpenAI()

    def embed_text(self, text: str) -> EmbeddingVector:
        return (
            self.client.embeddings.create(input=text, model=self.model)
            .data[0]
            .embedding
        )


class OllamaEmbedder(BaseEmbedder):
    """Ollama embedding model implementation."""

    def __init__(self, model: Literal["nomic-embed-text"], section_chunker=None):
        super().__init__(model, section_chunker)

    def embed_text(self, text: str) -> EmbeddingVector:
        return ollama.embeddings(prompt=text, model=self.model).embedding


class AsyncOpenAIEmbedder(BaseEmbedder):
    """Asynchronous OpenAI embedding model implementation."""

    def __init__(
        self,
        model: Literal["text-embedding-3-small", "text-embedding-3-large"],
        section_chunker=None,
    ):
        super().__init__(model, section_chunker)
        self.client = AsyncOpenAI()

    async def embed_text(self, text: str) -> EmbeddingVector:
        result = await self.client.embeddings.create(input=text, model=self.model)
        return result.data[0].embedding

    async def __call__(self, section: Section) -> EmbeddingList:
        if self.section_chunker is None:
            return [await self.embed_text(section.markdown)]

        doc = Document(text=section.markdown)
        nodes = self.section_chunker.get_nodes_from_documents([doc])
        return [await self.embed_text(node.get_text()) for node in nodes]


class WebpageProcessor:
    """Handles webpage processing and embedding operations."""

    @staticmethod
    def cosine_similarity(a: EmbeddingVector, b: EmbeddingVector) -> float:
        if not a or not b:
            return float("-inf")
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def compute_distance(
        section_embedding: EmbeddingList,
        query_embedding: EmbeddingVector,
        power: int = 1,
    ) -> float:
        similarities = [
            WebpageProcessor.cosine_similarity(emb, query_embedding)
            for emb in section_embedding
        ]
        try:
            return (
                sum(sim**power for sim in similarities) / len(section_embedding)
            ) ** (1.0 / power)
        except ZeroDivisionError:
            return float("-inf")
        except ValueError:
            return float("-inf")

    @staticmethod
    def create_section_graph(sections: list[Section]) -> dict[Section, list[Section]]:
        """Creates an adjacency list representation of the section hierarchy."""
        if not sections:
            return {}

        memory = [sections[0]]
        edges = []

        for section in sections[1:]:
            if section.level > memory[-1].level:
                memory.append(section)
            else:
                memory = memory[: section.level]
                memory[-1] = section
            try:
                edges.append((memory[-2], memory[-1]))
            except IndexError:
                edges.append((Section(title="", content=""), memory[-1]))

        # Convert edge list to adjacency list
        adjacency = {}
        for parent, child in edges:
            if parent not in adjacency:
                adjacency[parent] = []
            adjacency[parent].append(child)

        return adjacency


async def webpage_extractor(
    url: str,
    filter_type: ContentFilterType | str = ContentFilterType.HYBRID,
    custom_filter_instruction: Optional[str] = None,
    custom_filter_provider: Optional[str] = None,
    bypass_cache: bool = False,
) -> WebPage | str:
    """
    Extracts and processes webpage content with specified content filter.

    Args:
        url: The URL to extract content from
        filter_type: Type of content filter to use
        custom_filter_instruction: Custom instruction for the filter (only used if type is 'custom')
        custom_filter_provider: Custom provider for the filter (only used if type is 'custom')
    """
    try:
        # Create content filter
        content_filter = ContentFilterFactory.create_filter(
            filter_type, custom_filter_instruction, custom_filter_provider
        )

        # Create run configuration with the specified filter
        run_config = CrawlerRunConfig(
            excluded_tags=["nav", "footer", "header"],
            exclude_external_links=True,
            markdown_generator=DefaultMarkdownGenerator(content_filter=content_filter),
            check_robots_txt=True,
            bypass_cache=bypass_cache,
            no_cache_read=bypass_cache,
            no_cache_write=bypass_cache,
        )

        async with AsyncWebCrawler(config=BROWSER_CONFIG) as crawler:
            result = await crawler.arun(
                url=url,
                config=run_config,
                bypass_cache=bypass_cache,
                no_cache_read=bypass_cache,
                no_cache_write=bypass_cache,
            )

            # Parse markdown into sections
            parser = MarkdownNodeParser()
            doc = Document(text=result.markdown)
            nodes = parser.get_nodes_from_documents([doc])

            sections = [
                Section(
                    title=node.get_text().split("\n")[0],
                    content="\n".join(node.get_text().split("\n")[1:]),
                )
                for node in nodes
            ]

            # Create webpage structure
            adjacency = WebpageProcessor.create_section_graph(sections)
            return WebPage(url=url, body=adjacency, original_md=result.markdown)

    except httpx.HTTPError as e:
        return f"Error: {e} for url: {url} this may be a pdf file or the url might be invalid"


if __name__ == "__main__":
    import asyncio
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.markdown import Markdown
    from llama_index.core.node_parser import SentenceSplitter
    import numpy as np
    from dotenv import load_dotenv

    load_dotenv()

    async def main():
        console = Console()

        while True:
            url = Prompt.ask("URL")
            if url.lower() == "exit":
                break

            # Allow user to select content filter type
            print("\nAvailable content filter types:")
            for filter_type in ContentFilterType:
                print(f"- {filter_type.value}")
            filter_type = Prompt.ask(
                "Select content filter type", default=ContentFilterType.HYBRID.value
            )

            webpage = await webpage_extractor(
                url, filter_type, custom_filter_provider="openai/gpt-4o-mini"
            )
            if isinstance(webpage, str):
                console.print(webpage)
                continue

            # Display original content
            console.print(Markdown(webpage.original_md))
            console.print(Markdown(f"# Extracted content from {url}\n---"))

            # Initialize embedder
            embedder = OllamaEmbedder(
                "nomic-embed-text",
                section_chunker=SentenceSplitter(chunk_size=150, chunk_overlap=40),
            )

            # Process sections
            embedded_sections = [
                (section, embedder(section)) for section in webpage.sections
            ]

            # Interactive query loop
            while True:
                query = Prompt.ask("What would you like to know?")
                if query.lower() == "exit":
                    break

                query_embedding = embedder.embed_text(query)

                # Display results sorted by similarity
                console.print(Markdown("# Top Matching Sections"))
                sorted_sections = sorted(
                    embedded_sections,
                    key=lambda x: WebpageProcessor.compute_distance(
                        x[1], query_embedding
                    ),
                    reverse=True,
                )

                for section, embedding in sorted_sections[:3]:
                    similarity = WebpageProcessor.compute_distance(
                        embedding, query_embedding
                    )
                    console.print(Markdown(f"# Similarity: {similarity:.4f}\n"))
                    console.print(Markdown(section.markdown))
                    console.print(Markdown("---"))

    asyncio.run(main())
