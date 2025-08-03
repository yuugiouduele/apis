from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import re

# Qdrantクライアントの初期化（ローカルやクラウド環境の接続情報に合わせてください）
client = QdrantClient(host="localhost", port=6333)

### 3. クエリパラメータ安全性チェック（サニタイズに相当）

def is_safe_query_param(param: str) -> bool:
    """
    簡易的に、パラメータに不正な記号や危険な文字列がないかをチェック。
    Qdrant自体はSQLインジェクションがないが、アプリ層での妥当性検証として例示。
    """
    dangerous_patterns = [r";", r"--", r"\bDROP\b", r"\bDELETE\b", r"\bUPDATE\b"]
    for p in dangerous_patterns:
        if re.search(p, param, re.IGNORECASE):
            return False
    return True

def test_sanitization():
    safe_param = "cat"
    unsafe_param = "cat; DROP TABLE users; --"

    assert is_safe_query_param(safe_param), "安全なパラメータがNGになっています"
    assert not is_safe_query_param(unsafe_param), "危険なパラメータが通ってしまっています"
    print("サニタイズチェック完了")

test_sanitization()

### 4. ベクトル検索クエリの条件妥当性チェック（例：必須フィールド, 前方一致っぽい条件のチェック）

def validate_qdrant_query(filter_condition: Filter) -> bool:
    """
    QdrantのFilterオブジェクトに対して、以下を簡易チェック
      - フィルタ条件が空でないこと
      - 少なくとも1つのFieldConditionが present なこと
      - 前方一致的な条件（例えば、matchの部分一致）を含むこと（Qdrantは部分一致が基本ではないので模擬例）

    filter_condition: QdrantのFilterインスタンス
    """
    if filter_condition.is_empty():
        return False

    # condition.partsに条件があるか（FieldConditionのリスト）
    # ここはQdrantの仕様によるので、Filterはmust/shouldなど複数条件のPossibilityあり
    # 例としてmust条件が1つ以上あるか
    if not filter_condition.must:
        return False

    # フィールド条件の前方一致を模擬的に検査（例えばmatch.valueでパターンチェック）
    for cond in filter_condition.must:
        if isinstance(cond, FieldCondition) and isinstance(cond.match, MatchValue):
            val = cond.match.match
            if isinstance(val, str) and val.endswith("*"):  # ワイルドカードで前方一致
                return True
    return False

def test_validate_qdrant_query():
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    # 有効例：mustに前方一致想定チェックあり
    filter_good = Filter(
        must = [
            FieldCondition(
                key="name",
                match=MatchValue(match="apple*")  # 前方一致を'*'で表現と仮定
            )
        ]
    )

    # 無効例：mustなし
    filter_bad1 = Filter(must=[])
    # 無効例：前方一致なし
    filter_bad2 = Filter(
        must = [
            FieldCondition(
                key="name",
                match=MatchValue(match="apple")
            )
        ]
    )

    assert validate_qdrant_query(filter_good), "有効フィルターがNG"
    assert not validate_qdrant_query(filter_bad1), "空条件がOKになっている"
    assert not validate_qdrant_query(filter_bad2), "前方一致なし条件がOKになっている"

    print("Qdrantクエリ条件妥当性テスト完了")

test_validate_qdrant_query()

### 5. Qdrant検索実行時に例外・データロック検知

def search_vectors_safely(collection_name: str, vector: list, filter_condition: Filter):
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=vector,
            query_filter=filter_condition,
            limit=10,
        )
        return results
    except Exception as e:
        # Qdrantは通常ロック等は発生しづらいですが、例外キャッチ例示
        msg = str(e).lower()
        if "lock" in msg or "timeout" in msg:
            print(f"Warning: データロックやタイムアウトの可能性検出: {e}")
            return None
        else:
            raise

def test_search_handling():
    collection = "test_collection"
    dummy_vector = [0.1]*128  # 128次元ベクトル想定
    dummy_filter = Filter(
        must=[FieldCondition(key="category", match=MatchValue(match="fruit*"))]
    )
    # 通常は存在しないコレクション名などでエラー想定
    try:
        res = search_vectors_safely(collection, dummy_vector, dummy_filter)
        print(f"検索結果: {res}")
    except Exception as e:
        print("予期せぬ例外:", e)

test_search_handling()
