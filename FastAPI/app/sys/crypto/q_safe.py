from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

# 1量子ビット回路
qc = QuantumCircuit(1,1)
qc.h(0)            # アダマールゲートで重ね合わせ生成
qc.measure(0,0)    # 測定

backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
result = job.result()
counts = result.get_counts()

print(counts)
plot_histogram(counts)
