import numpy as np
import time

# 固定矩阵的维数
M = 1200
N = 1000
K = 800

print("Calculating running time...")
A = np.random.rand(M, N).astype(np.float32)
B = np.random.rand(N, K).astype(np.float32)
C = np.zeros((M, K), dtype=np.float32)   # 初始化结果矩阵 C

# 执行矩阵乘法并测量时间
start_time = time.time()

# 矩阵乘法
for i in range(M):
    for j in range(K):
        for k in range(N):
            C[i, j] += A[i, k] * B[k, j]

end_time = time.time()

def print_matrix(matrix, name):
    print(f"{name}:")
    for i in range(2):
        row = " ".join(f"{matrix[i, j]:.4f}" for j in range(2))
        print(row + (" ..." if matrix.shape[1] > 2 else ""))
    if matrix.shape[0] > 2:
        print("...")

# 打印矩阵 A、B 和 C 以及执行时间
print_matrix(A, "Matrix A")
print()
print_matrix(B, "Matrix B")
print()
print_matrix(C, "Matrix C (Result of A * B)")
print()
print("Run time:", end_time - start_time, "seconds")
