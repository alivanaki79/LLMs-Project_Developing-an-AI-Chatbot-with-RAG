# Financial Research Chatbot Tool 

This project aims to develop a research tool specifically designed for the financial field, functioning as a chatbot. Given the increasing importance and integration of chatbots into various aspects of life, this tool is designed to enhance speed, performance, and accuracy in financial analysis. The chatbot continuously updates itself with new articles and analyses in the financial sector, providing concise summaries and important details through a simple Q&A interface. Additionally, the chatbot can provide sources for its answers, facilitating comprehensive financial research with minimal time investment.

## Motivation and Importance

- **Continuous Information Updates:** Keeping up with the latest financial information is challenging. Manually copying and pasting information into a language model like ChatGPT is impractical due to word limits in prompts.
- **Consolidated Answers:** Financial information is often scattered across multiple articles or websites. This chatbot aims to consolidate answers, saving the user from reading several articles or inputting full texts into language models.
- **Efficiency:** Manual steps for collecting and analyzing financial data are time-consuming and costly, especially on a large scale. An AI-powered chatbot assistant provides continuous updates and streamlined information retrieval.

## Features

- Input up to 3 URLs of financial news articles.
- Load and process content from the provided articles.
- Split the content into manageable chunks for processing.
- Create embeddings for the text chunks using OpenAI's embeddings.
- Build and save a FAISS index for fast information retrieval.
- Query the processed data using OpenAI's language model.
- Display answers with sources to the user queries.

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/alivanaki79/LLMs-Project_Developing-an-AI-Chatbot-with-RAG.git
    ```
    
2. **Navigate to the project directory:**
   ```sh
   cd LLMs-Project_Developing-an-AI-Chatbot-with-RAG
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up your `.env` file:**
    Create a file named `.env` in the root directory of your project and add your OpenAI API key:
    ```plaintext
    OPENAI_API_KEY=your_openai_api_key_here
    ```

## Usage

1. **Run the Streamlit app:**
    ```sh
    streamlit run main.py
    ```

2. **Interact with the app:**
    - Enter up to 3 URLs of financial news articles in the sidebar.
    - Click on the "Process URLs" button to process the articles.
    - Enter your query in the "Question" input box and get answers with sources.

## Project Structure

- `main.py`: The main script for the Streamlit app.
- `requirements.txt`: A file listing all the dependencies required for the project.
- `faiss_store_openai.pkl`: A pickle file to store the FAISS index.
- `.env`: A file for storing environment variables, particularly the OpenAI API key.

## Notice:
- Replace `your_openai_api_key_here` with your actual OpenAI API key in the `.env` file.
- Ensure the `requirements.txt` file lists all the necessary dependencies for your project.
