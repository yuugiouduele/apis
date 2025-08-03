import psutil

def system_metrics_check():
    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    process_count = len(psutil.pids())
    thread_count = sum(p.num_threads() for p in psutil.process_iter())

    inode_usage = 0
    for part in psutil.disk_partitions():
        if 'rw' in part.opts:
            stats = psutil.disk_usage(part.mountpoint)
            inode_usage += stats.used  # 簡易代替

    print(f"Memory used: {mem.percent}%")
    print(f"CPU usage: {cpu_percent}%")
    print(f"Process count: {process_count}")
    print(f"Thread count: {thread_count}")
    print(f"Disk usage (approx inode): {inode_usage}")

    # 水準判定（例）
    if mem.percent > 90:
        print("メモリ警告！")
    if cpu_percent > 80:
        print("CPU高負荷警告！")

system_metrics_check()
