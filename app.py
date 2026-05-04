'''Run the app with:
            streamlit run app.py
'''

import streamlit as st  # web app framework for building the UI
from src.document_processor import process_document  # our document processing engine



######################################## Connect to OpenAI ###########################################################

from openai import OpenAI  # OpenAI client for making API calls
import os  # for reading environment variables
from dotenv import load_dotenv  # for loading our .env file

load_dotenv()  # loads the API key from our .env file into the environment

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # creates the OpenAI client using our API key

###################################### Ask OpenAI Questions on the text #########################################################

def answer_question(extracted_text, question):
    # sends the document text and user question to OpenAI and returns an answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # fast and cheap model, perfect for testing
        messages=[
            {
                "role": "system",  # sets the behaviour of the AI
                "content": "You are a helpful assistant that answers questions about documents. Only answer based on the document content provided. If the answer is not in the document, say so."
            },
            {
                "role": "user",  # the actual question from the user
                "content": f"Document content:\n{extracted_text}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content  # extracts just the text response



####################################### Set up page config ###################################################

st.set_page_config(
    page_title="Document Intelligence App",  # sets the browser tab title
    page_icon="📄",  # sets the browser tab icon
    layout="wide"  # uses full width of the screen rather than narrow column
)

st.title("📄 Document Intelligence App")  # main heading displayed on the page
st.write("Upload a document to extract text and ask questions about it.")  # subtitle


uploaded_file = st.file_uploader(
    "Choose a file",  # label displayed above the upload button
    type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"]  # restricts uploads to supported file types
)


################################# Ask Q and Embed AI answer into the Streamlit UI ###################################


if uploaded_file is not None:  # only run the code below if a file has been uploaded
    st.success(f"File uploaded: {uploaded_file.name}")  # show a green success message with the filename
    
    with st.spinner("Extracting text from document..."):  # show a loading spinner while processing
        file_bytes = uploaded_file.read()  # read the file into bytes
        extracted_text = process_document(file_bytes, uploaded_file.name)  # pass to our processing engine
    
    st.subheader("Extracted Text")  # section heading
    st.text_area("", extracted_text, height=300)  # display extracted text in a scrollable box


    st.subheader("Ask a Question")  # section heading
    
    question = st.text_input("Type your question about the document here:")  # text input box
    
    if question:  # only run if the user has typed a question
        with st.spinner("Thinking..."):  # show spinner while waiting for OpenAI response
            answer = answer_question(extracted_text, question)  # call our Q&A function
        
        st.subheader("Answer")  # section heading
        st.write(answer)  # display the answer

    

