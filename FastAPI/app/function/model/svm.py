from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# データセットロード（Irisデータセット例）
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 学習用・評価用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVMモデル作成（RBFカーネル）
svm = SVC(kernel='rbf', gamma='scale')
svm.fit(X_train, y_train)  # 学習

# 予測と評価
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"SVM分類の精度: {acc:.4f}")