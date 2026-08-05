from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter




data = PyPDFLoader(r"D:\gen-ai-with-muhammad-javed\RAG\Document Loader\GRU.pdf")

docs = data.load()

splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)

splits = splitter.split_documents(docs)


print(len(splits))

for i, splits in enumerate(splits):
    print(f"Chunk {i+1}: {splits.page_content}")
