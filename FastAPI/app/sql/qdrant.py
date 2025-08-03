from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

client = QdrantClient("localhost", port=6333)

## collectionnam add lists
collection_name = "my_cosine_collection"
vector_size = 128  # ベクトルの次元数に合わせて変更

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=vector_size,
        distance=Distance.COSINE
    )
)