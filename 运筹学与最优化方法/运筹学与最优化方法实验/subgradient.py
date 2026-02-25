import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def generate_data(n_nodes=10, m=10, n=200, sparsity=5, noise_std=0.1):
    np.random.seed(42)
    x_true = np.zeros(n)
    idx = np.random.choice(n, sparsity, replace=False)
    x_true[idx] = np.random.randn(sparsity)
    A_list = [np.random.randn(m, n) for _ in range(n_nodes)]
    e_list = [noise_std * np.random.randn(m) for _ in range(n_nodes)]
    b_list = [A @ x_true + e for A, e in zip(A_list, e_list)]
    return A_list, b_list, x_true

def solve_with_cvxpy(A_list, b_list, p):
    n = A_list[0].shape[1]
    x = cp.Variable(n)
    obj = 0.5 * sum([cp.sum_squares(A @ x - b) for A, b in zip(A_list, b_list)]) + p * cp.norm1(x)
    cp.Problem(cp.Minimize(obj)).solve()
    return x.value

# 次梯度法求解
def subgradient_solver(A_list, b_list, p, x_true, x_opt, stepsize=1e-4, max_iter=1000):
    n = A_list[0].shape[1]
    x = np.zeros(n)    # 初始化解
    d_true = []
    d_opt = []

    for k in range(max_iter):
        # Compute subgradient
        grad = np.zeros(n)
        for A, b in zip(A_list, b_list):
            grad += A.T @ (A @ x - b)   # 梯度部分
        grad += p * np.sign(x)          # L1 正则项的次梯度

        x -= stepsize * grad            # 参数更新

        d_true.append(np.linalg.norm(x - x_true))   # 距离真值
        d_opt.append(np.linalg.norm(x - x_opt))     # 距离最优解

    return d_true, d_opt

p = 0.1
A_list, b_list, x_true = generate_data()
x_opt = solve_with_cvxpy(A_list, b_list, p)
d_sub_true, d_sub_opt = subgradient_solver(A_list, b_list, p, x_true, x_opt)

plt.plot(d_sub_true, label='Subgrad: Distance to x_true')
plt.plot(d_sub_opt, label='Subgrad: Distance to x_opt')
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.title(f'Subgradient Method (p={p})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Subgradient Method p={p}:")
print(f"最终迭代距离真值 = {d_sub_true[-1]:.4f}")
print(f"最终迭代距离最优解 = {d_sub_opt[-1]:.4f}")
