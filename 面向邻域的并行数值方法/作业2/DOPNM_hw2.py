import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
p_val = 1 / (pi**2 - 1)
q_val = 1
a, b = -1, 1
alpha = 0                # u(a)
beta = -pi * np.e        # u(b)
h_list = [0.20, 0.10, 0.05, 0.02]

# 右端函数 f(x)
def f(x):
    return (2 * pi / (pi**2 - 1)) * np.cos(pi * x) * np.exp(x)

def exact_u(x):
    return np.sin(pi * x) * np.exp(x)

def gauss_solve(A, b):
    return np.linalg.solve(A, b)

def jacobi(A, b, iterations=30):
    x = np.zeros_like(b)
    D = np.diag(A)
    R = A - np.diagflat(D)
    for _ in range(iterations):
        x = (b - np.dot(R, x)) / D
    return x

# 构建刚度矩阵和载荷向量
def assemble(h):
    N = int((b - a) / h)
    x = np.linspace(a, b, N + 1)
    A = np.zeros((N + 1, N + 1))
    rhs = np.zeros(N + 1)

    for i in range(1, N):
        A[i, i-1] = p_val / h
        A[i, i]   = -2 * p_val / h + q_val * h
        A[i, i+1] = p_val / h
        rhs[i] = h * f(x[i])

    # 边界条件处理
    A[0, 0] = 1
    rhs[0] = alpha
    A[N, N] = 1
    rhs[N] = exact_u(b)

    return A, rhs, x

for h in h_list:
    A, rhs, x = assemble(h)
    u_gauss = gauss_solve(A, rhs)
    u_jacobi = jacobi(A, rhs, iterations=30)
    u_true = exact_u(x)

    # Task 1：数值解图像
    plt.figure()
    plt.plot(x, u_true, '--', color='green', label='Exact')
    plt.plot(x, u_gauss, color='tab:blue', label='Gauss')
    plt.plot(x, u_jacobi, color='orange', label='Jacobi (30 iters)')
    plt.title(f"Finite Element Solution Comparison (h = {h:.2f})")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

# Task 2：误差图像
plt.figure()
for h in h_list:
    A, rhs, x = assemble(h)
    u_gauss = gauss_solve(A, rhs)
    u_jacobi = jacobi(A, rhs, iterations=30)
    u_true = exact_u(x)

    err_gauss = np.abs(u_true - u_gauss)
    err_jacobi = np.abs(u_true - u_jacobi)

    plt.plot(x, err_gauss, label=f"Gauss h={h}")
    plt.plot(x, err_jacobi, '--', label=f"Jacobi h={h}")

plt.title("Error Comparison")
plt.xlabel("x")
plt.ylabel("Absolute Error |u(x) - u_h(x)|")
plt.legend(loc='upper left', fontsize=9, frameon=True)
plt.grid(True)
plt.tight_layout()
plt.show()