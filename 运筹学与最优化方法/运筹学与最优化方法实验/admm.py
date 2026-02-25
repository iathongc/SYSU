import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def generate_data(n_nodes=10, m=10, n=200, sparsity=5, noise_std=0.1):
    np.random.seed(42)
    x_true = np.zeros(n)
    idx = np.random.choice(n, sparsity, replace=False)
    x_true[idx] = np.random.randn(sparsity)                             # 构造稀疏向量
    A_list = [np.random.randn(m, n) for _ in range(n_nodes)]            # 每个节点的观测矩阵
    e_list = [noise_std * np.random.randn(m) for _ in range(n_nodes)]   # 噪声
    b_list = [A @ x_true + e for A, e in zip(A_list, e_list)]           # 带噪声的观测
    return A_list, b_list, x_true

def solve_with_cvxpy(A_list, b_list, p):
    n = A_list[0].shape[1]
    x = cp.Variable(n)
    obj = 0.5 * sum([cp.sum_squares(A @ x - b) for A, b in zip(A_list, b_list)]) + p * cp.norm1(x)
    cp.Problem(cp.Minimize(obj)).solve()
    return x.value

def admm_solver(A_list, b_list, p, x_true, x_opt, rho=1.0, max_iter=1000):
    n_nodes = len(A_list)
    n = A_list[0].shape[1]

    # 初始化
    x_list = [np.zeros(n) for _ in range(n_nodes)]    # 局部变量
    z = np.zeros(n)                                   # 全局变量
    u_list = [np.zeros(n) for _ in range(n_nodes)]    # 对偶变量

    d_true = []     # 距离 x_true 的误差
    d_opt = []      # 距离最优解 x_opt 的误差

    # 提前计算每个节点需要用到的逆矩阵
    inv_matrices = [np.linalg.inv(A.T @ A + rho * np.eye(n)) for A in A_list]

    for _ in range(max_iter):
        for i in range(n_nodes):    # 更新局部变量 x_i
            A, b = A_list[i], b_list[i]
            u = u_list[i]
            rhs = A.T @ b + rho * (z - u)
            x_list[i] = inv_matrices[i] @ rhs

        # 更新共享变量 z
        x_u_avg = np.mean([x_list[i] + u_list[i] for i in range(n_nodes)], axis=0)
        z = np.sign(x_u_avg) * np.maximum(np.abs(x_u_avg) - p / (n_nodes * rho), 0)

        for i in range(n_nodes):          #更新对偶变量 u_i
            u_list[i] += x_list[i] - z

        # 记录每次迭代的误差
        x_avg = np.mean(x_list, axis=0)
        d_true.append(np.linalg.norm(x_avg - x_true))
        d_opt.append(np.linalg.norm(x_avg - x_opt))

    return d_true, d_opt

p = 0.1
A_list, b_list, x_true = generate_data()
x_opt = solve_with_cvxpy(A_list, b_list, p)
d_admm_true, d_admm_opt = admm_solver(A_list, b_list, p, x_true, x_opt)

plt.plot(d_admm_true, label='ADMM: Distance to x_true')
plt.plot(d_admm_opt, label='ADMM: Distance to x_opt')
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.title(f'ADMM Method (p={p})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"ADMM Method p={p}:")
print(f"最终迭代距离真值 = {d_admm_true[-1]:.4f}")
print(f"最终迭代距离最优解 = {d_admm_opt[-1]:.4f}")
