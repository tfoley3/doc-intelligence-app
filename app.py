'''Run the app with:
            streamlit run app.py
'''

import streamlit as st  # web app framework for building the UI
from src.document_processor import process_document  # our document processing engine
from src.rag_processor import chunk_text, embed_and_store, retrieve_relevant_chunks # RAG functions
from src.ab_testing import run_ab_test # A/B testing function


######################################## Connect to OpenAI ###########################################################

from openai import OpenAI  # OpenAI client for making API calls
import os  # for reading environment variables
from dotenv import load_dotenv  # for loading our .env file

load_dotenv()  # loads the API key from our .env file into the environment

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # creates the OpenAI client using our API key

###################################### Ask OpenAI Questions on the text #########################################################

def answer_question(extracted_text, question):
    # sends the document text and user question to OpenAI and returns an answer
    try:
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
    except Exception as e: 
        return f"Sorry, I couldn't get an answer from OpenAI: {str(e)}" # graceful failure
    

def summarise_document(extracted_text):
    # sends the document to OpenAI and asks for a concise summary
    try: 
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
    except Exception as e:
        return f"Sorry, I couldn't generate a summary: {str(e)}" # graceful failure



####################################### Set up page config ###################################################

st.set_page_config(
    page_title="Document Intelligence App",  # sets the browser tab title
    page_icon="📄",  # sets the browser tab icon
    layout="wide"  # uses full width of the screen rather than narrow column
)

st.title("📄 Document Intelligence App")  # main heading displayed on the page
st.write("Upload a document to extract text and ask questions about it.")  # subtitle


uploaded_files = st.file_uploader(
    "Choose one or more files",  # label displayed above the upload button
    type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],  # restricts uploads to supported file types
    accept_multiple_files=True # enables multiple file selection
)


################################# Ask Q and Embed AI answer into the Streamlit UI ###################################


if uploaded_files:  # only run the code below if at least one file has been uploaded
    # check each file size - reject anything over 10MB 
    oversized = [f.name for f in uploaded_files if f.size > 10 * 1024 * 1024]
    if oversized:
        st.error(f"These filesa re too large (max 10MB): {', '.join(oversized)}")
    else:
        st.success(f"{len(uploaded_files)} file(s) uploaded: {', '.join([f.name for f in uploaded_files])}")

    try:
        with st.spinner("Extracting text from all documents..."):
            all_text = {}  # dictionary to store text from each file
            for f in uploaded_files:  # loop through each uploaded file
                file_bytes = f.read()  # read the file into bytes
                text = process_document(file_bytes, f.name)  # extract text
                if text.strip():  # only add if text was actually extracted
                    all_text[f.name] = text  # store with filename as key
                else:
                    st.warning(f"Could not extract text from {f.name} — skipping.")
        
        if not all_text:  # if no text was extracted from any file
            st.error("No text could be extracted from any of the uploaded documents.")
            st.stop()
        
        # combine all text for display and Q&A
        extracted_text = "\n\n---\n\n".join(all_text.values())  # join with separator
        
    except Exception as e:
        st.error(f"Something went wrong reading the documents: {str(e)}")
        st.stop() 


    st.subheader("Extracted Text")
    
    # show each document's text in its own expander
    for filename, text in all_text.items():
        with st.expander(f"📄 {filename}"):  # collapsible section per document
            st.text_area("", text, height=200, key=filename)  # unique key per file
    
    # only chunk and embed if the set of documents has changed
    current_files = sorted([f.name for f in uploaded_files])  # sorted list of filenames
    if "rag_processed" not in st.session_state or st.session_state.get("processed_files") != current_files:
        with st.spinner("Processing all documents for RAG..."):
            chunks = chunk_text(extracted_text)  # chunk the combined text
            embed_and_store(chunks)
            st.session_state.rag_processed = True
            st.session_state.chunk_count = len(chunks)
            st.session_state.processed_files = current_files  # remember which files were processed
    
    st.success(f"{len(all_text)} document(s) processed into {st.session_state.chunk_count} chunks for RAG search")

    if st.button("Summarise Document(s)"): #creates a clickable button
        for filename, text in all_text.items(): #loop through each document
            with st.spinner(f"Summarising {filename}..."): #spinner while waiting for OpenAI
                try:
                    summary = summarise_document(text) #call our summarise function
                    st.subheader(f"Summary - {filename}") #section heading 
                    st.write(summary) # display the summary
                except Exception as e:
                    st.error(f"Could not summarise {filename}: {str(e)}")


    st.subheader("A/B Model Comparison")  # section heading
    st.write("Test two models on the same question and compare results.")  # description
    
    ab_question = st.text_input(
        "Enter a question to test across models:",  # label
        key="ab_question"  # unique key needed since we have two text inputs
    )
    
    if st.button("Run A/B Test"):  # button to trigger the test
        if ab_question:  # only run if a question has been entered
            with st.spinner("Running A/B test across models..."):
                ab_context = retrieve_relevant_chunks(ab_question)  # get relevant chunks only
                results = run_ab_test(ab_question, ab_context)  # run the test
            
            # display results side by side
            col1, col2, col3 = st.columns(3)  # create three columns
            
            with col1:  # left column — gpt-4o-mini results
                st.markdown("### gpt-4o-mini")
                st.write(results["gpt-4o-mini"]["answer"])
                st.metric("Response time", f"{results['gpt-4o-mini']['response_time']}s")
                st.metric("Tokens used", results["gpt-4o-mini"]["token_count"])
            
            with col2:  # middle column — gpt-3.5-turbo results
                st.markdown("### gpt-3.5-turbo")
                st.write(results["gpt-3.5-turbo"]["answer"])
                st.metric("Response time", f"{results['gpt-3.5-turbo']['response_time']}s")
                st.metric("Tokens used", results["gpt-3.5-turbo"]["token_count"])
            
            with col3:  # right column — distilgpt2 (local)
                st.markdown("### DistilGPT-2 (local)")
                st.write(results["distilgpt2"]["answer"])
                st.metric("Response time", f"{results['distilgpt2']['response_time']}s")
                st.metric("Cost", "Free - runs locally")

            st.success("Results logged to MLflow!")
        else:
            st.warning("Please enter a question first.")  # prompt if no question entered
    
    
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

    

