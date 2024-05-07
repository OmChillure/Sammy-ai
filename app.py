import os
import google.generativeai as genai
from google.generativeai import generate_embeddings
import streamlit as st
from  PyPDF2 import PdfReader
# from flask import Flask, render_template, request
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
api = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api)


def get_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectors(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def convo_chain():
  prompt_template = """
        Based on the given text, Generate the following types of question and their corresponding answers:
        1.Give the abstract from the pdf(less than 50 words)
        2.Brief answer questions (3 questions and their answers)

        Text:{context}
  """
  model = ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.3)
  prompt = PromptTemplate(template=prompt_template, input_variables=['context','question'])
  chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
  return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = convo_chain()
    response = chain({"input_documents":docs , "question": user_question},return_only_outputs=True)
    print(response)
    st.write("Replay", response['output_text'])



st.set_page_config("Chat PDF")
st.header("Chat With Sammy")
user_question = "Create Questions Answers:"
if st.button("Generate Q&A"):
    user_input(user_question)
with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files Here", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vectors(text_chunks)
            st.success("Done")
            