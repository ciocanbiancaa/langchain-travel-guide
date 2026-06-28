import os

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = Ollama(model = "llama3.2")

title_prompt = PromptTemplate(
    input_variables = ["topic"],
    template= """You are an experienced speech writer.
    You need to craft an impactful title for a speech 
    on the following topic: {topic}
    Answer exactly with one title."""

)

speech_prompt = PromptTemplate(
    input_variables = ["title"],
    template= """You need to write a powerful speech of 350 words
    for the following title: {title}"""


)


chain_1 = title_prompt | llm | StrOutputParser()
chain_2 = speech_prompt | llm
final_chain = chain_1 | chain_2

st.title("Speech Generator")

topic = st.text_input("Enter the topic: ")


if topic:
    response = final_chain.invoke({"topic": topic})

    st.write(response)

