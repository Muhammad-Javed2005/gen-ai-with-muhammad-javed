from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings


CHROMA_PATH = "./chroma_db"


def get_vector_store():

    embeddings = MistralAIEmbeddings(
        model="mistral-embed"
    )

    vector_store = Chroma(
        collection_name="deep_learning_pdf",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vector_store