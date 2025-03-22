import ollama 
from database import Chroma
    
    
def ollama_context(query,store):
    retrieved_data=Chroma.query_chroma(query,store=store)
    
    if not retrieved_data or "documents" not in retrieved_data or not retrieved_data["documents"]:
        return "No relevant context found in the database. Please provide more information."

    
    
    context = "\n".join(retrieved_data["documents"][0])if retrieved_data["documents"] else ""
    prompt = f"Context:\n{context}\n\nUser Query: {query}\nAnswer: should be strictly from context you should not provide and other data just present the context in a personalized way short and crisp "
    
    response= ollama.chat(model="llama3.2:1b",messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

if __name__ == "__main__":
    
    for i in range(1000):   
     update_database=str(input("1 for not updateing and 0 for updating : ")) 
       
     
     user_query = input("user : ")
     response = ollama_context(user_query,update_database)
     print("Ollama RAG Response:", response)
