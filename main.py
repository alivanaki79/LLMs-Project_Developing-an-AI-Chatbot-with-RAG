import os
import streamlit as st
import pickle
import time
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file, specifically the OpenAI API key

st.title("AI Chatbot with RAG")
st.sidebar.title(" URLs in the field of finance")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")  # Collect up to 3 URLs from the user via the sidebar
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")  # Button to start processing the entered URLs
file_path = "faiss_store_openai.pkl"  # File path for saving the FAISS index

main_placeholder = st.empty()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, max_tokens=500)  # Initialize the OpenAI chat model with specified parameters

if process_url_clicked:
    # Step 1: Load data from the provided URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...")
    data = loader.load()

    # Step 2: Split the loaded data into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],  # Define separators to split the text
        chunk_size=1000  # Specify the maximum size of each chunk
    )
    main_placeholder.text("Text Splitting...Started...")
    docs = text_splitter.split_documents(data)  # Split the data into chunks

    # Step 3: Create embeddings from the text chunks and build the FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Building Embedding Vector...")
    time.sleep(2)  # Short delay to simulate processing time

    # Step 4: Save the FAISS index to a file using pickle
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

# Query input and processing section
query = main_placeholder.text_input("Please write your question in the box below and then press enter:")
if query:
    if os.path.exists(file_path):  # Check if the FAISS index file exists
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)  # Load the FAISS index from the file
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)  # Perform the query using the retrieval chain

        # Display the answer and the sources of the information
        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")  # Get the sources, if any
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline for display
            for source in sources_list:
                st.write(source)
