from langchain_community.document_loaders import TextLoader
import os
from langchain_community.document_loaders import WebBaseLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

ctfcollection_name="cookm-chroma"

embeddings=DashScopeEmbeddings(dashscope_api_key=os.environ.get("OPENAI_API_KEY"),model="text-embedding-v2" )

vectorstore = Chroma(
        collection_name=ctfcollection_name,
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_qwen_db",
    )

def init():
    urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=700, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore.add_documents(documents=doc_splits)

def save_vector(paths):

    docs = [TextLoader(path).load() for path in paths]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB


    vectorstore.add_documents(documents=doc_splits)

    return vectorstore
