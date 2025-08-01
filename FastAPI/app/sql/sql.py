from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

# Qdrantサーバへ接続（Docker Composeのサービス名・ポートに合わせてください）
qdrant = QdrantClient(host="qdrant", port=6333)

COLLECTION_NAME = "items_collection"

def create_collection():
    if not qdrant.get_collection(collection_name=COLLECTION_NAME):
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE)
        )

def add_item_vector(item_id: int, vector: np.ndarray, payload: dict):
    # QdrantはVectorとMetadataのペアで保存
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(id=item_id, vector=vector.tolist(), payload=payload)
        ]
    )

def search_similar_vectors(query_vector: np.ndarray, limit: int = 5):
    result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector.tolist(),
        limit=limit,
        with_payload=True,
    )
    return result
