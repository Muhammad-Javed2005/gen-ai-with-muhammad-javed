from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
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


llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0
)


retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 3}
    ),
    llm=llm
)


question = "What is gradient descent?"


docs = retriever.invoke(question)


print("\n===== MULTI QUERY RETRIEVER =====\n")

for i, doc in enumerate(docs, 1):

    print(f"--- Document {i} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
    print()