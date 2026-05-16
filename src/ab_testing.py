import mlflow # experiment tracking
import time # for measuring response time
from openai import OpenAI # OpenAI client
import os # for environment variables
from dotenv import load_dotenv # for loading .env file
from src.huggingface_model import answer_question_hf # HuggingFace QA model

load_dotenv() # load API key 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # initialise OpenAI client 

# Now setting up the MLflow experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # use SQLite instead of file storage
mlflow.set_experiment("document_qa_comparison") # creates experiment called document_qa_comparison 

def run_ab_test(question, context):
    # tests two models on the same question and logs results to MLflow
    models = ["gpt-4o-mini", "gpt-3.5-turbo", "distilgpt2"] # models to compare
    results = {} # dictionary to store results for each model

    for model in models: # loop through each model
        with mlflow.start_run(run_name=model): # start a new MLflow run for this model

            start_time = time.time() # record start time

            # send the question Locally or to OpenAI
            if model == "distilgpt2":
                # use HuggingFace locally - no API call needed
                answer = answer_question_hf(context, question)
                token_count = 0 # local model has no token cost
            else:
                # use OpenAI API
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions about documents. Only answer based on the document content provided."
                        },
                        {
                            "role": "user",
                            "content": f"Document content:\n{context}\n\nQuestion: {question}"
                        }
                    ]
                )
                answer = response.choices[0].message.content # extract the answer
                token_count = response.usage.total_tokens # total tokens used

            end_time = time.time() # record end time
            response_time = round(end_time - start_time, 2) # calculate duration in seconds

            # log everything to MLflow
            mlflow.log_param("model", model)             # which model was used
            mlflow.log_param("question", question)       # what was asked
            mlflow.log_metric("response_time", response_time)   # how long it took 
            mlflow.log_metric("token_count", token_count)       # tokens used
            mlflow.log_text(answer, "answer.txt")               # the actual answer

            results[model] = {
                "answer": answer,
                "response_time": response_time,
                "token_count": token_count
            }
    return results # return results for both models
