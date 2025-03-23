import ollama 
from database import Chroma
from fastapi import FastAPI
from pydantic import BaseModel
import requests

app=FastAPI()

class QueryRequest(BaseModel):
    query: str
    store: str 
    language : str

def translate_mymemory(text, source_lang="en", target_lang="gu"):
    url = "https://api.mymemory.translated.net/get"
    params = {
        "q": text,
        "langpair": f"{source_lang}|{target_lang}"
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, params=params,headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return data.get("responseData", {}).get("translatedText", "Translation failed")
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def ollama_context(query,store):
    retrieved_data=Chroma.query_chroma(query,store=store)
    
    if not retrieved_data or "documents" not in retrieved_data or not retrieved_data["documents"]:
        return "No relevant context found in the database. Please provide more information."
    
    context = "\n".join(retrieved_data["documents"][0])if retrieved_data["documents"] else ""
    
    
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
    print(response)
    return response["message"]["content"]

@app.post("/query/")
async def query_ollama(request:QueryRequest):
    print(request)
    response_text =  ollama_context(request.query,request.store)
    
    print(response_text)
    if request.language.lower() == "hindi":
        translated_text = translate_mymemory(response_text,target_lang="hi")
        print(translated_text)
        return {"response": translated_text}

    elif request.language.lower() == "gujarati":
        translated_text =translate_mymemory(response_text,target_lang="gu")
        print(translated_text)
        return {"response": translated_text}

    return {"response": response_text}
    
 
    
    