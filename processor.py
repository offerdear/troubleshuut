import fitz  # PyMuPDF
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_chunks_from_manual(file_bytes: bytes, filename: str, product_id: str = None):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    chunks_with_metadata = []
    chapter, section = None, None

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")

        match = re.search(r"Chapter\s+\d+:\s+([^\n]+)", page_text)
        if match:
            chapter = match.group(1).strip()

        headings = re.findall(r"\n([A-Z][A-Za-z ,]{3,50})\n", page_text)
        if headings:
            section = headings[0].strip()

        chunks = text_splitter.split_text(page_text)
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": filename,
                "product_id": product_id or filename.split('.')[0],
                "page": page_num + 1,
                "chapter": chapter or "Unknown",
                "section": section or "Unknown",
                "chunk_id": f"{filename}_{page_num+1}_{i}"
            }
            chunks_with_metadata.append((chunk, metadata))

    return chunks_with_metadata
