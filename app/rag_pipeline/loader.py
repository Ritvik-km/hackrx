# app/rag_pipeline/loader.py
import tempfile
import aiohttp
from typing import List

from langchain_text_splitters import SpacyTextSplitter
from app.config import settings

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import spacy
import re

async def download_pdf_to_tempfile(url: str) -> str:
    """Download PDF from URL and save to a temp file"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    async with aiohttp.ClientSession() as sess:
        async with sess.get(url) as resp:
            async for chunk in resp.content.iter_chunked(8192):
                tmp.write(chunk)
    tmp.close()
    return tmp.name

def smart_split_insurance_document(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    """
    Custom splitter for insurance documents that preserves definition integrity
    """
    # Split by section numbers first (like 3.22, 10.16, etc.)
    section_pattern = r'(\n\d+\.\d+\.?\s+[A-Z][^.\n]*(?:\s+means|\s+refers|\s+includes|\s+shall))'
    sections = re.split(section_pattern, text)
    
    chunks = []
    current_chunk = ""
    
    for i, section in enumerate(sections):
        if not section.strip():
            continue
            
        # If adding this section would exceed chunk size, save current chunk
        if len(current_chunk) + len(section) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous
            if chunk_overlap > 0:
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                current_chunk = overlap_text + section
            else:
                current_chunk = section
        else:
            current_chunk += section
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If no sections found, fall back to regular splitting
    if len(chunks) <= 1:
        text_splitter = SpacyTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            pipeline = "en_core_web_sm",
            # separators=["\n\n", "\n", ". "]
        )
        return text_splitter.split_text(text)
    
    # Post-process: ensure no chunk exceeds maximum safe size (6000 chars ≈ 1500 tokens)
    final_chunks = []
    max_safe_size = 6000
    
    for chunk in chunks:
        if len(chunk) <= max_safe_size:
            final_chunks.append(chunk)
        else:
            # Split oversized chunks using recursive splitter
            text_splitter = SpacyTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                pipeline = "en_core_web_sm",
                # separators=["\n\n", "\n", ". "]
            )
            sub_chunks = text_splitter.split_text(chunk)
            final_chunks.extend(sub_chunks)
    
    return final_chunks

async def load_and_chunk_pdf(url: str) -> List[Document]:
    """Load PDF from URL and chunk it into documents"""
    # Download PDF to temp file
    temp_path = await download_pdf_to_tempfile(url)
    
    try:
        # Load PDF
        loader = PyPDFLoader(temp_path)
        raw_documents = loader.load()
        
        # Combine all pages into one text for better section detection
        full_text = "\n".join([doc.page_content for doc in raw_documents])
        
        # Use custom smart splitting
        chunk_texts = smart_split_insurance_document(
            full_text, 
            chunk_size=settings.CHUNK_SIZE, 
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        
        # Convert back to Document objects
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            # Find which page this chunk primarily belongs to
            page_num = 0
            char_count = 0
            for j, raw_doc in enumerate(raw_documents):
                if char_count + len(raw_doc.page_content) >= len(chunk_text) * 0.5:
                    page_num = j
                    break
                char_count += len(raw_doc.page_content)
            
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "chunk_index": i,
                    "source": temp_path,
                    "page": page_num
                }
            )
            chunks.append(doc)
        
        return chunks
        
    finally:
        # Clean up temp file
        import os
        if os.path.exists(temp_path):
            os.unlink(temp_path)