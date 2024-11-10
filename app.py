import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize session state for API key and chat history
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_embeddings():
    if st.session_state.openai_api_key:
        return OpenAIEmbeddings(
            openai_api_key=st.session_state.openai_api_key,
            allowed_special={'<|endofprompt|>'}
        )
    return None

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    embeddings = initialize_embeddings()
    if embeddings:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_db")
    else:
        st.error("Please enter a valid OpenAI API key first.")

def get_conversational_chain(tools, ques):
    if not st.session_state.openai_api_key:
        st.error("Please enter your OpenAI API key first.")
        return
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", 
        temperature=0, 
        openai_api_key=st.session_state.openai_api_key
    )
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful assistant. Answer the question as detailed as possible from the provided context. If the answer is not in
    provided context, just say, "Answer is not available in the context." Do not provide an incorrect answer."""
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    tool = [tools]
    agent = create_tool_calling_agent(llm, tool, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    
    try:
        response = agent_executor.invoke({"input": ques})
        return response['output']
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def user_input(user_question):
    if not st.session_state.openai_api_key:
        st.error("Please enter your OpenAI API key first.")
        return
        
    embeddings = initialize_embeddings()
    if embeddings:
        try:
            new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
            retriever = new_db.as_retriever()
            retrieval_chain = create_retriever_tool(
                retriever,
                "pdf_extractor",
                "This tool answers queries from the PDF."
            )
            response = get_conversational_chain(retrieval_chain, user_question)
            return response
        except Exception as e:
            st.error(f"An error occurred: {str(e)}. Make sure you've processed PDF files first.")
    else:
        st.error("Please enter a valid OpenAI API key first.")

def display_chat_history():
    # Create a container for the chat history
    chat_container = st.container()
    
    # Display all messages in the chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").markdown(
                    f"<div style='color: black;'>{message['message']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.chat_message("assistant").markdown(
                    f"<div style='color: black;'>{message['message']}</div>",
                    unsafe_allow_html=True
                )

def display_api_key_message():
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px'>
            <h4 style='color: #008080; margin-top: 0;'>📝 Need an OpenAI API Key?</h4>
            <p>To use this app, you'll need an OpenAI API key. You can get a free API key by following these steps:</p>
            <ol>
                <li>Visit <a href='https://platform.openai.com/api-keys' target='_blank'>OpenAI's API Key Page</a></li>
                <li>Sign up or log in to your OpenAI account</li>
                <li>Click on "Create new secret key"</li>
                <li>Copy your API key and paste it in the sidebar</li>
            </ol>
            <p><em>Note: Keep your API key secure and never share it publicly!</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    # Set up main page configuration
    st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #FF5733;'>✨Interactive PDF Chat Tool</h1>", unsafe_allow_html=True)
    st.write("---")
    
    # Sidebar styling and configuration
    with st.sidebar:
        st.title("Configuration")
        st.markdown("<h3 style='color: #008080;'>Enter Your OpenAI API Key:</h3>", unsafe_allow_html=True)
        api_key = st.text_input("API Key:", type="password", placeholder="sk-...")
        if api_key:
            st.session_state.openai_api_key = api_key
            st.success("API key set successfully!")

        st.markdown("<h3 style='color: #008080;'>Upload Your PDF Files:</h3>", unsafe_allow_html=True)
        pdf_doc = st.file_uploader("Choose PDF files", accept_multiple_files=True)
        if st.button("Embed", help="Processes the uploaded PDFs for queries"):
            if not st.session_state.openai_api_key:
                st.error("Please enter your OpenAI API key first.")
            elif not pdf_doc:
                st.error("Please upload PDF files first.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = pdf_read(pdf_doc)
                    text_chunks = get_chunks(raw_text)
                    vector_store(text_chunks)
                    st.success("PDF Processing Complete!")

        # Add a clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Display API key message if not set
    if not st.session_state.openai_api_key:
        display_api_key_message()

    # Main interaction section
    st.markdown("<h2 style='color: #FFA500;'>Ask Questions from PDF Content:</h2>", unsafe_allow_html=True)
    st.write("Type your question in the chat below to get answers from the uploaded PDF content.")

    # Display existing chat history
    display_chat_history()

    # Handle new user input
    user_question = st.chat_input("Type your question here...")
    
    if user_question:
        if not st.session_state.openai_api_key:
            st.error("Please enter an OpenAI API key in the sidebar before asking questions.")
            display_api_key_message()
        else:
            # Get the assistant's response
            response = user_input(user_question)
            
            if response:
                # Append the new conversation to the history
                st.session_state.chat_history.append({"role": "user", "message": user_question})
                st.session_state.chat_history.append({"role": "assistant", "message": response})
                
                # Rerun the app to update the display
                st.rerun()

    # Footer with additional information
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #5D6D7E;'>"
        "© 2024 PDF Chat Tool - Simplifying Document Conversations"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()