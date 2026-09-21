from langchain_text_splitter import RecursiveCharacterTextSplitter
from typing import List, List, list
from langchain_core.documents import Document

class EnterpriseChunker:
    """
    Enterprise-grade text chunking strategy using LangChain's RecursiveCharacterTextSplitter.
    to preserve semantic context and avoid splitting sentence abrutly
    """


    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initializes the chunker with specified chunk size and overlap.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Takes raw loaded documents and splits them into smaller, manageable chunks.
        """
        if not documents:
            raise ValueError("No documents provided for splitting.")
            
        chunks = self.splitter.split_documents(documents)
        return chunks


    def split_text(self, text: str) -> List[str]:
        """
        Takes raw string text and splits it into text chunks.
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Provided text is empty.")
            
        return self.splitter.split_text(text)