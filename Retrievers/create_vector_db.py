from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "retriever_demo"


docs = [
    Document(
        page_content="Gradient descent is an optimization algorithm used in machine learning.",
        metadata={"topic": "gradient_descent"}
    ),

    Document(
        page_content="Gradient descent minimizes the loss function.",
        metadata={"topic": "gradient_descent"}
    ),

    Document(
        page_content="Gradient descent is an optimization method that minimizes the loss function.",
        metadata={"topic": "gradient_descent"}
    ),

    Document(
        page_content="Neural networks use gradient descent for training.",
        metadata={"topic": "neural_network"}
    ),

    Document(
        page_content="Support Vector Machines are supervised learning algorithms.",
        metadata={"topic": "svm"}
    ),

    Document(
        page_content="Neural networks learn patterns from training data using optimization algorithms.",
        metadata={"topic": "neural_network"}
    ),

    Document(
        page_content="The learning rate controls how large each gradient descent update is.",
        metadata={"topic": "gradient_descent"}
    ),

    Document(
        page_content="A loss function measures the difference between predicted and actual values.",
        metadata={"topic": "loss_function"}
    ),
]


embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)



vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)


# Avoid inserting documents again if database already contains data
if vectorstore._collection.count() == 0:

    vectorstore.add_documents(docs)

    print("Documents embedded and saved to ChromaDB.")

else:

    print("ChromaDB already contains documents.")
    print(f"Documents: {vectorstore._collection.count()}")