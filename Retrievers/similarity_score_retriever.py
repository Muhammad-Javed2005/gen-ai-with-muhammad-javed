from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "retriever_demo"


embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)


question = "What is gradient descent?"


results = vectorstore.similarity_search_with_score(
    question,
    k=3
)


print("\n===== SIMILARITY SEARCH WITH SCORE =====\n")


for i, (doc, score) in enumerate(results, 1):

    print(f"--- Document {i} ---")
    print("Score:", score)
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print()