import threading

lock = threading.Lock()
shared_counter = 0

def add():
    global shared_counter
    for _ in range(100000):
        with lock:  # この範囲が排他制御
            shared_counter += 1

threads = [threading.Thread(target=add) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(shared_counter)  # 正しく排他できれば200000
