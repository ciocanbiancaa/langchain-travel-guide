

import os

from langchain_classic import text_splitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model = "nomic-embed-text")

document = TextLoader("job_listings.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=10)
chunks = text_splitter.split_documents(document)
db = Chroma.from_documents(chunks, embeddings)
retriever = db.as_retriever()



text = input("enter the query ")


docs = retriever.invoke(text)


#print(docs)

print(docs[0].page_content)

'''for doc in docs:
    print(doc.page_content)'''