import os

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser


llm1 = Ollama(model = "llama3.2")
llm2 = Ollama(model = "mistral")
title_prompt = PromptTemplate(
    input_variables = ["topic"],
    template= """You are an experienced speech writer.
    You need to craft an impactful title for a speech 
    on the following topic: {topic}
    Answer exactly with one title."""

)

speech_prompt = PromptTemplate(
    input_variables = ["title", "emotion"],
    template= """You need to write a powerful {emotion} speech of 350 words
    for the following title: {title}
    Format the output with 2 keys: 'tile', 'speech' and fill them
    with the respective values
    """


)


chain_1 = title_prompt | llm1 | StrOutputParser() | (lambda title: {"title" : title, "emotion": emotion})
chain_2 = speech_prompt | llm2 | JsonOutputParser() 
final_chain = chain_1 | chain_2

st.title("Speech Generator")

topic = st.text_input("Enter the topic: ")
emotion = st.text_input("Enter the emotion: ")


if topic:
    response = final_chain.invoke({"topic": topic})

    st.write(response)

