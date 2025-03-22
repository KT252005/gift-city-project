import ollama 
from database import Chroma
from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()
class QueryRequest(BaseModel):
    query: str
    store: str 

def ollama_context(query,store):
    retrieved_data=Chroma.query_chroma(query,store=store)
    
    if not retrieved_data or "documents" not in retrieved_data or not retrieved_data["documents"]:
        return "No relevant context found in the database. Please provide more information."

    
    
    context = "\n".join(retrieved_data["documents"][0])if retrieved_data["documents"] else ""
    
    print(context)
    prompt = f"""
    
    Context:\n{context}\n\n
    User Query: {query}\n\n  
    Instructions:
    - Answer **strictly** using the provided context.
    - **Do not** provide any information that is not found in the context.
    - **If the context is empty or does not contain relevant information, say:**
     "I couldn't find relevant information in the provided data."
    - Keep the answer **concise and to the point**.
    - Do not assume or infer missing details outside the given context.
    \nAnswer:"""
    
    response= ollama.chat(model="llama3.2:1b",messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

@app.post("/query/")
async def query_ollama(request:QueryRequest):
    response_text =ollama_context(request.query,request.store)
    return {"response": response_text}
