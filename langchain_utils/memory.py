import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import PointStruct
from dotenv import load_dotenv
load_dotenv()

# Configuration
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "blog-fact-validator"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Initialize Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30
)

# Lazy-load embedding model
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder

# Ensure collection exists (idempotent)
def create_collection_if_not_exists():
    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
create_collection_if_not_exists()

# Store a fact into Qdrant memory
def store_fact(fact_text: str, metadata: dict):
    try:
        embedder = get_embedder()
        vector = embedder.encode(fact_text).tolist()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, fact_text))  # Stable UUID based on fact content

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={**metadata, "fact": fact_text}
                )
            ]
        )
    except Exception as e:
        print(f"[ERROR] Failed to store fact in Qdrant: {e}")

# Query memory for relevant facts
def retrieve_memory_facts(query: str, top_k: int = 5) -> list:
    try:
        embedder = get_embedder()
        vector = embedder.encode(query).tolist()
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=top_k
        )
        return [r.payload for r in results if r.payload]
    except Exception as e:
        print(f"[ERROR] Failed to retrieve facts from Qdrant: {e}")
        return []

# For dev/testing
def reset_memory():
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        create_collection_if_not_exists()
        print("[INFO] Memory reset successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to reset memory: {e}")
