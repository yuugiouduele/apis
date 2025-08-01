from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from qdrant_client import models
from qdrant_client import QdrantClient

# Qdrant部分を同ファイルにまとめるか、別モジュールで管理してください
COLLECTION_NAME = "items_collection"

app = FastAPI()

# Qdrantクライアント設定
qdrant = QdrantClient(host="qdrant", port=6333)

# ベクトルサイズは利用モデルに依存（ここは128次元を仮定）
VECTOR_SIZE = 128

# 埋め込みモデル初期化（文書やクエリのベクトル化に使用）
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# データモデル
class Item(BaseModel):
    id: int
    name: str
    description: Optional[str] = None

class SearchQuery(BaseModel):
    query: str
    limit: Optional[int] = 5

# 起動時にコレクション作成
@app.on_event("startup")
def on_startup():
    if not qdrant.get_collection(collection_name=COLLECTION_NAME):
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
        )

# アイテム登録（ベクトル化してQdrantに保存）
@app.post("/items/", response_model=Item)
def create_item(item: Item):
    vector = embedding_model.encode(item.description or item.name)
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(id=item.id, vector=vector.tolist(), payload=item.dict())
        ]
    )
    return item

# ベクトル検索API（クエリ文からベクトル化して検索）
@app.post("/search/")
def search_items(search: SearchQuery):
    query_vector = embedding_model.encode(search.query)
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector.tolist(),
        limit=search.limit,
        with_payload=True,
    )
    # 結果からペイロードのみ取り出し
    found_items = [hit.payload for hit in results]
    return {"results": found_items}

# ここに通常の CRUD API も加えることが可能です。
