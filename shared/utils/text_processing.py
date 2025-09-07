import re
from typing import List, Dict, Any
from pathlib import Path
import aiofiles
from pypdf import PdfReader
from docx import Document as DocxDocument


class TextExtractor:
    """Handles text extraction from various file formats."""

    @staticmethod
    async def extract_from_file(file_path: str) -> str:
        """Extract text from a file based on its extension."""
        file_extension = Path(file_path).suffix.lower()

        if file_extension == ".pdf":
            return await TextExtractor._extract_pdf(file_path)
        elif file_extension in [".docx", ".doc"]:
            return await TextExtractor._extract_docx(file_path)
        elif file_extension == ".txt":
            return await TextExtractor._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    @staticmethod
    async def _extract_pdf(file_path: str) -> str:
        """Extract text from PDF file."""
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

    @staticmethod
    async def _extract_docx(file_path: str) -> str:
        """Extract text from DOCX file."""
        doc = DocxDocument(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    @staticmethod
    async def _extract_txt(file_path: str) -> str:
        """Extract text from TXT file."""
        async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
            return await file.read()


class TextCleaner:
    """Handles text cleaning and preprocessing."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace
        text = text.strip()
        # Normalize line breaks
        text = re.sub(r"\n+", "\n", text)
        return text

    @staticmethod
    def remove_special_characters(text: str, keep_newlines: bool = True) -> str:
        """Remove special characters while optionally keeping newlines."""
        if keep_newlines:
            # Keep newlines, remove other special chars
            text = re.sub(r"[^\w\s\n]", "", text)
        else:
            # Remove all special characters
            text = re.sub(r"[^\w\s]", "", text)
        return text


class DocumentChunker:
    """Handles document chunking with sliding window approach."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize chunker with configurable size and overlap.

        Args:
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            # Find the end of the chunk
            end = start + self.chunk_size

            # If we're not at the end, try to find a good break point
            if end < len(text):
                # Look for sentence endings within the last 100 chars of the chunk
                last_period = text.rfind(".", start, end)
                last_newline = text.rfind("\n", start, end)

                # Use the latest sentence ending or newline
                break_point = max(last_period, last_newline)
                if break_point > start + self.chunk_size - 100:
                    end = break_point + 1
                else:
                    end = start + self.chunk_size

            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - self.overlap

            # Ensure we don't get stuck
            if start >= len(text) or end <= start:
                break

        return chunks

    def chunk_by_sentences(self, text: str, max_sentences: int = 5) -> List[str]:
        """Chunk text by sentences."""
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if adding this sentence would exceed chunk size
            if current_length + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)

        # Add the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class MetadataExtractor:
    """Extracts metadata from documents."""

    @staticmethod
    def extract_metadata(file_path: str) -> Dict[str, Any]:
        """Extract basic metadata from file."""
        path = Path(file_path)
        stat = path.stat()

        metadata = {
            "filename": path.name,
            "file_size": stat.st_size,
            "file_extension": path.suffix,
            "created_at": stat.st_ctime,
            "modified_at": stat.st_mtime,
        }

        return metadata

    @staticmethod
    def extract_text_metadata(text: str) -> Dict[str, Any]:
        """Extract metadata from text content."""
        lines = text.split("\n")
        word_count = len(text.split())
        char_count = len(text)
        line_count = len(lines)

        # Estimate reading time (200 words per minute)
        reading_time_minutes = word_count / 200

        metadata = {
            "word_count": word_count,
            "character_count": char_count,
            "line_count": line_count,
            "estimated_reading_time": reading_time_minutes,
        }

        return metadata


# Convenience functions
async def process_document(
    file_path: str, chunk_size: int = 1000, overlap: int = 200
) -> Dict[str, Any]:
    """Complete document processing pipeline."""
    # Extract text
    raw_text = await TextExtractor.extract_from_file(file_path)

    # Clean text
    cleaned_text = TextCleaner.clean_text(raw_text)

    # Extract metadata
    file_metadata = MetadataExtractor.extract_metadata(file_path)
    text_metadata = MetadataExtractor.extract_text_metadata(cleaned_text)

    # Chunk text
    chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk_text(cleaned_text)

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "chunks": chunks,
        "metadata": {**file_metadata, **text_metadata},
    }
