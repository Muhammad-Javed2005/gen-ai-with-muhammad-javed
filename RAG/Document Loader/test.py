from langchain_community.document_loaders import TextLoader


data = TextLoader(r"D:\gen-ai-with-muhammad-javed\RAG\Document Loader\notes.txt",  encoding="utf-8")

docs = data.load()

print(docs)


