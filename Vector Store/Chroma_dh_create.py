from langchain_chroma import Chroma 
from langchain_mistralai import MistralAIEmbeddings


CHROMA_PATH = "./chroma_db"

def get_embeddings():
    return MistralAIEmbeddings(
        model = "mistral-embed"
    )


def get_vector_store():
    embeddings = get_embeddings


    vector_store = Chroma(
        collection_name="deeplearning.pdf",
        embedding_function= embeddings,
        persist_directory=CHROMA_PATH
    )


    return vector_store


