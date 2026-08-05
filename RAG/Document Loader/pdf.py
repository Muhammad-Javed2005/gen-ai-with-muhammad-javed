from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader(r"D:\gen-ai-with-muhammad-javed\RAG\Document Loader\GRU.pdf")



docs = data.load()

print(docs)

