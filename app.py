'''Run the app with:
            streamlit run app.py
'''

import streamlit as st  # web app framework for building the UI
from src.document_processor import process_document  # our document processing engine
from src.rag_processor import chunk_text, embed_and_store, retrieve_relevant_chunks # RAG functions


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

def summarise_document(extracted_text):
    # sends the document to OpenAI and asks for a concise summary
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # same model as our Q&A function
        messages=[
            {
                "role": "system",  # instructs the AI how to behave
                "content": "You are a helpful assistant that summarises documents. Provide a clear and concise summary with the key points highlighted."
            },
            {
                "role": "user",  # the actual request
                "content": f"Please summarise the following document:\n\n{extracted_text}"
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

    # Process document for RAG as soon as it is uploaded
    # But only chunk and embed if we haven't already done it for this document
    if "rag_processed" not in st.session_state:
        with st.spinner("Processing document for RAG..."):  # spinner while chunking and embedding 
            chunks = chunk_text(extracted_text)             # split into chunks
            embed_and_store(chunks)                         # embed and store in ChromaDB
            st.session_state.rag_processed = True # flag so we don't repeat this
            st.session_state.chunk_count = len(chunks) # store chunk count for display
            
        st.success(f"Document processed into {len(chunks)} chunks from RAG search") # confirmation message


    if st.button("Summarise Document"): #creates a clickable button
        with st.spinner("Summarising..."): #spinner while waiting for OpenAI
            summary = summarise_document(extracted_text) #call our summarise function
        st.subheader("Summary") #section heading 
        st.write(summary) # display the summary

    
    
    # initialise chat history in session state if it doesn't exist yet
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.subheader("Ask a Question")  # section heading

    # Toggle between basic Q&A and RAG 
    use_rag = st.toggle(
        "Use RAG (recommended for large documents)", # label
        value=True # default to RAG on 
    )

    # display all previous messages in the conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  # renders as user or assistant bubble
            st.write(message["content"])  # display the message content

    # chat input box at the bottom
    question = st.chat_input("Type your question about the document here...")

    if question:  # only run if the user has typed a question
        # add user question to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):  # display user message bubble
            st.write(question)

        with st.spinner("Thinking..."):  # spinner while waiting for OpenAI
            if use_rag: # use RAG if toggle is on
                context = retrieve_relevant_chunks(question) # get relevant chunks 
                answer = answer_question(context, question) # answer only using relevant chunks
            else: # use basic Q&A if toggle is off
                answer = answer_question(extracted_text, question)  # call our Q&A function
        
        # add AI answer to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        with st.chat_message("assistant"):  # display assistant message bubble
            st.write(answer)  # display the answer

    

