import numpy as np
import matplotlib.pyplot as plt

# 解析解函数 u(x)
def u_exact(x):
    return np.sin(np.pi * x) * np.exp(x)

# 右端函数 f(x)
def f_rhs(x):
    return (2 * np.pi / (np.pi**2 - 1)) * np.cos(np.pi * x) * np.exp(x)

# 构建差分方程组
def construct_system(h):
    a, b = -1.0, 1.0
    alpha = 0                   # 左边界 Dirichlet 条件
    beta = -np.pi * np.exp(1)   # 右边界 Neumann 条件

    N = int((b - a) / h)        # 网格节点数
    x = np.linspace(a, b, N + 1)  # 均匀划分的网格点

    p = -1 / (np.pi**2 - 1)
    q = 1

    A = np.zeros((N+1, N+1))    # 系数矩阵初始化
    F = np.zeros(N+1)           # 右端向量初始化

    # 内部点使用中心差分
    for i in range(1, N):
        A[i, i-1] = p / h**2
        A[i, i]   = -2 * p / h**2 + q
        A[i, i+1] = p / h**2
        F[i] = f_rhs(x[i])

    # 左端点 Dirichlet 边界条件
    A[0, 0] = 1
    F[0] = alpha

    # 右端点 Neumann 边界条件
    A[N, N] = 1 / h
    A[N, N-1] = -1 / h
    F[N] = beta

    return A, F, x

# 高斯消元法
def gauss_elimination(A, b):
    return np.linalg.solve(A, b)

# 雅可比迭代法
def jacobi(A, b, x0=None, max_iter=30):
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    D = np.diag(np.diag(A))     # 提取对角线矩阵 D
    R = A - D                   # 剩余部分 R = A - D
    D_inv = np.linalg.inv(D)    # 计算 D 的逆
    x = x0.copy()
    for _ in range(max_iter):
        x = np.dot(D_inv, b - np.dot(R, x))  # 迭代公式
    return x

hs = [0.20, 0.10, 0.05, 0.02]

for h in hs:
    A, F, x = construct_system(h)
    u_true = u_exact(x)

    # 求数值解
    u_gs = gauss_elimination(A, F)
    u_jacobi = jacobi(A, F)

    # Task 1：绘制数值解
    plt.figure()
    plt.plot(x, u_true, label="Exact", linestyle='--')
    plt.plot(x, u_gs, label="Gauss")
    plt.plot(x, u_jacobi, label="Jacobi", linestyle=':')
    plt.title(f"Solution Comparison (h = {h:.2f})")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid()

    # Task 2：绘制误差图
    error_gs = u_gs - u_true
    error_jacobi = u_jacobi - u_true

    plt.figure()
    plt.plot(x, error_gs, label=f"Gauss error (h = {h:.2f})")
    plt.plot(x, error_jacobi, label=f"Jacobi error (h = {h:.2f})", linestyle='--')
    plt.axhline(0, color='gray', linestyle=':')  # 参考线 y = 0
    plt.title(f"Error Curves (h = {h:.2f})")
    plt.xlabel("x")
    plt.ylabel("Error: $u_h - u(x)$")
    plt.legend()
    plt.grid()

# 显示所有图像
plt.show()
