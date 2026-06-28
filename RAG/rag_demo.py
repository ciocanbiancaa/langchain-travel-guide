

import os

from langchain_classic import text_splitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


embeddings = OllamaEmbeddings(model = "nomic-embed-text")

llm = Ollama(model = "llama3.2")

prompt_template = ChatPromptTemplate.from_messages(
[
    ("system","""You are an assistant for answering questions.
    Use the provided context to respond.If the answer 
    isn't clear, acknowledge that you don't know. 
    Limit your response to three concise sentences.
    {context}
    
    """),
    ("human", "{input}")
]

)


document = TextLoader("product-data.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=20)
chunks = text_splitter.split_documents(document)
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

qa_chain =  create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, qa_chain)

print("chat with document")
question = input("your question")

if question:
    response = rag_chain.invoke({"input":question})
    print(response['answer'])

