

import os

from langchain_community.embeddings import OllamaEmbeddings
import numpy as np
embeddings = OllamaEmbeddings(model = "llama3.2")

text1 = input("enter the text_1 ")
text2 = input("enter the text_2")

response1 = embeddings.embed_query(text1)
response2 = embeddings.embed_query(text2)

similarity_score = np.dot(response1, response2) / (np.linalg.norm(response1) * np.linalg.norm(response2))

print(similarity_score*100, "%")