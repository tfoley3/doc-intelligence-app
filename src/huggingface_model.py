from transformers import pipeline  # HuggingFace pipeline for easy model loading
import os  # for environment variables

def answer_question_hf(context, question):
    # loads a pre-trained DistilBERT model fine-tuned for question answering
    qa_pipeline = pipeline(
        "text-generation",          # task type - supported in all versions
        model="distilgpt2",         # small, fast pre-trained model - runs locally
        max_new_tokens=100          # limit response length
    )
    
    prompt = f"Context: {context[:500]}\n\nQuestion: {question}\n\nAnswer:"

    result = qa_pipeline(
        prompt,  
        do_sample=False     # deterministic output - no randomness
    )
    
    # extract just the generated answer after "Answer:"
    full_output = result[0]["generated_text"]
    answer = full_output.split("Answer:")[-1].strip() 

    return answer if answer else "No answer found." 


