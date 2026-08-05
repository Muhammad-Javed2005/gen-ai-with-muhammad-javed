from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader


splitter = CharacterTextSplitter(
    separator= "",
    chunk_size=10,
    chunk_overlap=1
    
)

data  = TextLoader(r"D:\gen-ai-with-muhammad-javed\RAG\Document Loader\notes.txt", encoding="utf-8")


docs = data.load()


print(len(docs))
print(docs)

chunks = splitter.split_documents(docs)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.page_content}")

