import threading
import time

event = threading.Event()

def worker():
    print("ワーカー: 開始待ち")
    event.wait()  # イベントがセットされるまで待つ
    print("ワーカー: 処理開始")

th = threading.Thread(target=worker)
th.start()

time.sleep(1)
print("メイン: ワーカー開始を通知")
event.set()  # ここでworkerが再開
th.join()
