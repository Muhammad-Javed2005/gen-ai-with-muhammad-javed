from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chat_models import init_chat_model
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_embeddings import HuggingFaceEmbeddings
# from 



load_dotenv()

data = PyPDFLoader(r"D:\gen-ai-with-muhammad-javed\RAG\Document Loader\GRU.pdf")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docs = data.load()

spilltter = RecursiveCharacterTextSplitter(
    chunks_size = 1000 , 
    chunk_overlap = 120 ,

)

split = spilltter.split_documents(docs)


model = ChatMistralAI(
    model = "mistral-small-2506", 
    callbacks= None , 
    temperature = 0.9
)


model = invoke("How can be help in the cycle....")
result = model.inoke