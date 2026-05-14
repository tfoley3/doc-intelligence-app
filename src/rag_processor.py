from langchain_text_splitters import RecursiveCharacterTextSplitter  # splits text into chunks
from langchain_openai import OpenAIEmbeddings  # converts text to vectors using OpenAI
import chromadb  # vector database for storing and searching embeddings
import os  # for reading environment variables
from dotenv import load_dotenv  # for loading our .env file

load_dotenv()  # load the API key


# initialise the OpenAI embeddings model
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")  # load API key from .env file
)

# initialise ChromaDB client and create a collection to store our vectors
chroma_client = chromadb.Client()  # creates an in-memory ChromaDB instance
collection = chroma_client.get_or_create_collection(
    name="documents"  # name of our vector store collection
)

def chunk_text(text):
    #splits the document text into overlapping chunks 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # each chunk is max 500 characters
        chunk_overlap=50, # chunks overlap by 50 characters to preserve context at boundaries
        separators=["\n\n", "\n", ".", " "] # tries to split on these in order 
    )
    chunks = splitter.split_text(text) # returns a list of text chunks
    return chunks 

def embed_and_store(chunks):
    # converts chunks to vectors and stores them in ChromaDB
    for i, chunk in enumerate(chunks): #loop through each chunk with an index number 
        vector = embeddings.embed_query(chunk) # convert chunk to a vector
        collection.add(
            ids=[str(i)],           # unique ID for each chunk
            embeddings=[vector],    # the vector representation
            documents=[chunk]       # the original text chunk
        )


def retrieve_relevant_chunks(question, n_results=3):
    # converts the question to a vector and finds the most similar chunks
    question_vector = embeddings.embed_query(question)  # embed the question
    
    results = collection.query(
        query_embeddings=[question_vector],  # search using the question vector
        n_results=n_results                  # return the top 3 most similar chunks
    )
    
    relevant_text = "\n\n".join(results["documents"][0])  # join the chunks into one string
    return relevant_text  # return just the most relevant sections
