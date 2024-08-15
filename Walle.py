import streamlit as st
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import logging
from langchain_community.chat_models import ChatOllama
from config import GOOGLE_API_KEY
import os
import multiprocessing
from tqdm import tqdm

# Set up Google API Key for Google Palm
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

# Create a logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Function to load a subset of product data
def load_product_data(file_path, nrows=10000):
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        return df.to_string(index=False)
    except Exception as e:
        logger.error(f"Error loading product data: {str(e)}")
        st.error("Failed to load product data. Please check the file and try again.")
        return None

# Function to split text into chunks
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to process chunks in parallel
def process_chunk(chunk):
    return chunk

# Function to create a vector store with multiprocessing
def get_vector_store(text_chunks):
    try:
        embeddings = GooglePalmEmbeddings()
        with multiprocessing.Pool() as pool:
            processed_chunks = list(tqdm(pool.imap(process_chunk, text_chunks), total=len(text_chunks), desc="Processing chunks"))
        vector_store = FAISS.from_texts(processed_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        st.error("Failed to create vector store. Please try again.")
        return None

# Function to create a conversational chain using Ollama with 
def get_conversational_chain_offline(vector_store):
    try:
        sol_model = ChatOllama(model="walle")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=sol_model,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        logger.error(f"Error creating conversational chain: {str(e)}")
        st.error("Failed to initialize the AI model. Please try again.")
        return None

# Function to handle user input
def user_input(user_question, conversation):
    try:
        response = conversation({'question': user_question})
        return response['chat_history']  # Return the updated chat history
    except Exception as e:
        logger.exception("Error in user input processing")
        st.error(f"An error occurred: {str(e)}")
        return []

def main():
    st.set_page_config(page_title="Wall-E: Walmart's AI Shopping Assistant ", layout="wide")
    st.header("Wall-E: Your Walmart Shopping Assistant 🛒")
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        st.session_state.chat_history = []
        
        # Load and process the product data
        with st.spinner("Loading and processing product data..."):
            product_data = load_product_data("Apple.csv")
            if product_data:
                text_chunks = get_text_chunks(product_data)
                vector_store = get_vector_store(text_chunks)
                if vector_store:
                    st.session_state.conversation = get_conversational_chain_offline(vector_store)
                    st.success("Product data loaded successfully. You can start chatting now!")
                else:
                    st.error("Failed to process the product data. Please check the file and restart the application.")

    # Main chat interface
    user_question = st.chat_input("What are you looking for today?")
    
    if user_question:
        if st.session_state.conversation:
            st.session_state.chat_history = user_input(user_question, st.session_state.conversation)  # Update chat history
        else:
            st.warning("The AI model is not initialized. Please check the product data and restart the application.")

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)

if __name__ == "__main__":
    main()