import multiprocessing
import time

def worker(counter, lock):
    for _ in range(1000):
        with lock:
            counter.value += 1

def test_data_race_condition():
    counter = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()

    procs = [multiprocessing.Process(target=worker, args=(counter, lock)) for _ in range(10)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    expected = 10 * 1000
    print(f"counter={counter.value}, expected={expected}")
    assert counter.value == expected, "データ競合が発生している可能性あり"

test_data_race_condition()
