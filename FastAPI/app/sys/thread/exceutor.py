from concurrent.futures import ThreadPoolExecutor

def task(x):
    print(f"処理: {x}")
    return x * x

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(task, range(3)))

print(results)  # [0, 1, 4]
