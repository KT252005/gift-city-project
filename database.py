import json
import chromadb
from sentence_transformers import SentenceTransformer

class Chroma:
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient("chroma_data/chroma_db")
    collection = chroma_client.get_or_create_collection(name="my_collection")
    
    # Load embedding model
    embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    
    # Sample documents
    documents = [
    {"id": "0", "text": "This is a document about pineapple"},
    {"id": "1", "text": "This is a document about oranges"},
    {"id": "2", "text": "Hi I am Kanak, what are you doing?"},
    {"id": "3", "text": "Kanak is brilliant"},
    {"id": "4", "text": "Kanak is dumb"},
    {"id": "5", "text": "Space is beautiful"},
    {"id": "6", "text": "The sun is a star at the center of our solar system"},
    {"id": "7", "text": "The theory of relativity was developed by Albert Einstein"},
    {"id": "8", "text": "Python is a popular programming language for machine learning"},
    {"id": "9", "text": "Neural networks are a key component of deep learning"},
    {"id": "10", "text": "Blockchain technology enables secure and decentralized transactions"},
    {"id": "11", "text": "Elon Musk founded SpaceX to make space travel more accessible"},
    {"id": "12", "text": "The Eiffel Tower is located in Paris, France"},
    {"id": "13", "text": "Mount Everest is the highest mountain in the world"},
    {"id": "14", "text": "Water boils at 100 degrees Celsius at sea level"},
    {"id": "15", "text": "The mitochondria is the powerhouse of the cell"},
    {"id": "16", "text": "Artificial Intelligence is revolutionizing many industries"},
    {"id": "17", "text": "The Amazon rainforest is the largest tropical rainforest on Earth"},
    {"id": "18", "text": "Photosynthesis is the process by which plants convert sunlight into energy"},
    {"id": "19", "text": "The speed of light is approximately 299,792,458 meters per second"},
    {"id": "20", "text": "The Great Wall of China is a famous historical structure"},
    {"id": "21", "text": "Quantum mechanics describes the behavior of particles on a microscopic scale"},
    {"id": "22", "text": "Saturn is known for its prominent ring system"},
    {"id": "23", "text": "Cloud computing allows remote access to computing resources"},
    {"id": "24", "text": "The human brain contains approximately 86 billion neurons"},
    {"id": "25", "text": "The Wright brothers invented the first successful airplane"},
    {"id": "26", "text": "Renewable energy sources include solar, wind, and hydro power"},
    {"id": "27", "text": "Leonardo da Vinci was a Renaissance artist and inventor"},
    {"id": "28", "text": "The Pacific Ocean is the largest ocean on Earth"},
    {"id": "29", "text": "Antarctica is the coldest continent on the planet"},
    {"id": "30", "text": "Jupiter is the largest planet in our solar system"},
    {"id": "31", "text": "Machine learning algorithms improve with more data"},
    {"id": "32", "text": "The Moon orbits the Earth approximately every 27 days"},
    {"id": "33", "text": "The Internet has transformed communication and access to information"},
    {"id": "34", "text": "Shakespeare wrote the play 'Hamlet'"},
    {"id": "35", "text": "Cryptography is used to secure digital communications"},
    {"id": "36", "text": "The Pyramids of Giza are one of the Seven Wonders of the Ancient World"},
    {"id": "37", "text": "The Hubble Space Telescope has provided incredible images of the universe"},
    {"id": "38", "text": "The Industrial Revolution began in the 18th century"},
    {"id": "39", "text": "DNA carries genetic instructions for living organisms"},
    {"id": "40", "text": "The Mars Rover is exploring the surface of Mars"},
    {"id": "41", "text": "The Black Hole is a region of spacetime with extremely strong gravity"},
    {"id": "42", "text": "Nanotechnology is revolutionizing medicine and materials science"},
    {"id": "43", "text": "A tsunami is a series of ocean waves caused by underwater earthquakes"},
    {"id": "44", "text": "Big Data analytics helps businesses make data-driven decisions"},
    {"id": "45", "text": "Hurricanes form over warm ocean waters"},
    {"id": "46", "text": "Carbon dioxide is a greenhouse gas that contributes to global warming"},
    {"id": "47", "text": "Programming languages like C, Java, and Python are widely used in software development"},
    {"id": "48", "text": "Tesla is a company that focuses on electric vehicles and renewable energy"},
    {"id": "49", "text": "The human heart pumps blood throughout the body"},
    {"id": "50", "text": "Quantum computing could revolutionize information processing"}
]

    @staticmethod
    def precompute_embeddings():
        """Compute embeddings for all documents and save to JSON file"""
        embeddings = {doc["id"]: Chroma.embedding_model.encode(doc["text"]).tolist() 
                     for doc in Chroma.documents}
        
        with open("embeddings.json", "w") as f:
            json.dump(embeddings, f)
        
        print("Precomputed embeddings saved!")
        return embeddings
    
    @staticmethod
    def store():
        """Load precomputed embeddings and store documents in ChromaDB"""
        try:
            with open("embeddings.json", "r") as f:
                precomputed_embeddings = json.load(f)
        except FileNotFoundError:
            # If embeddings file doesn't exist, create it
            precomputed_embeddings = Chroma.precompute_embeddings()
        
        # Store documents with precomputed embeddings in ChromaDB
        Chroma.collection.upsert(
            ids=[doc["id"] for doc in Chroma.documents],
            embeddings=[precomputed_embeddings[doc["id"]] for doc in Chroma.documents],
            documents=[doc["text"] for doc in Chroma.documents]
        )
        
        print("Documents stored in ChromaDB with precomputed embeddings")
    
    @staticmethod
    def query_chroma(query_text, store):
        if store == "0":
            Chroma.precompute_embeddings()
            return None
        else:
            Chroma.store()
            query_embedding = Chroma.embedding_model.encode(query_text).tolist()

            results = Chroma.collection.query(
                query_embeddings=[query_embedding],
                n_results=5  # Number of results to return
            )
            return results


