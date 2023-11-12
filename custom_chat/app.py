import os
import requests
import pickle
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import WebBaseLoader


def save_file(raw_text):
    try:
        with open("history/history.txt", "w", encoding="utf-8") as file:
            file.write(raw_text)
        st.success("Saved successfully.")
    except Exception as er:
        st.warning(f"Error: {er}. No file to save.")


def load_file():
    try:
        uploaded_file = st.file_uploader("Choose a file", type="txt")
        if uploaded_file is not None:
            load = uploaded_file.read().decode("utf-8")
            st.success("File loaded successfully.")
            return load
        else:
            st.warning("Please choose a valid file.")
    except Exception as e:
        st.warning(f"Error loading file: {e}")


def close_chat():
    st.session_state.conversation = None
    st.session_state.chat_history = None
    st.success("Chat closed.")


def chat(raw_text):
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    st.session_state.conversation = get_conversation_chain(vectorstore)


def get_html_text(new_doc):
    text = ""
    for doc in new_doc:
        text += doc.page_content
    return text


def get_pdf_text(new_doc):
    text = ""
    for pdf in new_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_docx_text(new_doc):
    text = ""
    for docx in new_doc:
        doc = Document(docx)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    return text


def get_txt_text(new_doc):
    text = ""
    for txt_doc in new_doc:
        text += txt_doc.getvalue().decode('utf-8') + "\n"
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        pass


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.button("Create new chat")
    with col2:
        if st.button("Save file"):
            save_file()
    with col3:
        if st.button("Load file"):
            load = load_file()
            if load is not None:
                st.write(load)
    with col4:
        st.button("Close Chat", on_click=close_chat)

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        choice = st.radio("Choose an option:", ("Upload PDF file",
                                                "Upload TXT file",
                                                "Upload DOCX file",
                                                "Enter web link", ))
        try:
            if choice == "Enter web link":
                new_doc = st.text_input("Enter a web link:")
                if st.button("Process Web Link"):
                    loader = WebBaseLoader(web_path=new_doc)
                    html_doc = loader.load()
                    with st.spinner("Processing"):
                        raw_text = get_html_text(html_doc)
                        chat(raw_text)

            elif choice == "Upload PDF file":
                new_doc = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
                if st.button("Process"):
                    with st.spinner("Processing"):
                        raw_text = get_pdf_text(new_doc)
                        chat(raw_text)

            elif choice == "Upload TXT file":
                new_doc = st.file_uploader("Upload your TXTs here and click on 'Process'", accept_multiple_files=True)
                if st.button("Process"):
                    with st.spinner("Processing"):
                        raw_text = get_txt_text(new_doc)
                        chat(raw_text)

            elif choice == "Upload DOCX file":
                new_doc = st.file_uploader("Upload your DOCXs here and click on 'Process'", accept_multiple_files=True)
                if st.button("Process"):
                    with st.spinner("Processing"):
                        raw_text = get_docx_text(new_doc)
                        chat(raw_text)

        except Exception as ex:
            st.error(f"Error input!")


if __name__ == '__main__':
    main()