import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def generate_data(n_nodes=10, m=10, n=200, sparsity=5, noise_std=0.1):
    np.random.seed(42)     # 设置随机种子
    x_true = np.zeros(n)   # 初始化稀疏信号
    idx = np.random.choice(n, sparsity, replace=False)     # 随机选择非零元素的位置
    x_true[idx] = np.random.randn(sparsity)                # 给稀疏向量赋随机值
    A_list = [np.random.randn(m, n) for _ in range(n_nodes)]
    e_list = [noise_std * np.random.randn(m) for _ in range(n_nodes)]
    b_list = [A @ x_true + e for A, e in zip(A_list, e_list)]   
    return A_list, b_list, x_true 

# 使用 cvxpy 求解 L1 正则化最小二乘问题，作为参考最优解
def solve_with_cvxpy(A_list, b_list, p):
    n = A_list[0].shape[1]     # 变量维度
    x = cp.Variable(n)         # 定义优化变量 x

    # 构造目标函数，包括误差项和 L1 正则化
    obj = 0.5 * sum([cp.sum_squares(A @ x - b) for A, b in zip(A_list, b_list)]) + p * cp.norm1(x)

    # 构造并求解优化问题
    cp.Problem(cp.Minimize(obj)).solve()
    return x.value

def prox_L1(v, tau):
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0)

def proximal_gradient(A_list, b_list, p, x_true, x_opt, stepsize=1e-3, max_iter=1000):
    n = A_list[0].shape[1]      # 变量维度
    x = np.zeros(n)             # 初始解为零
    d_true = [] 
    d_opt = []            

    def grad(x):        # 计算目标函数的梯度
        return sum([A.T @ (A @ x - b) for A, b in zip(A_list, b_list)])

    for _ in range(max_iter):
        x = prox_L1(x - stepsize * grad(x), stepsize * p)    # 计算梯度并进行更新

        # 记录误差
        d_true.append(np.linalg.norm(x - x_true))
        d_opt.append(np.linalg.norm(x - x_opt))
    
    return d_true, d_opt

p = 0.1
A_list, b_list, x_true = generate_data()
x_opt = solve_with_cvxpy(A_list, b_list, p)
d_pg_true, d_pg_opt = proximal_gradient(A_list, b_list, p, x_true, x_opt)

plt.plot(d_pg_true, label='PG: Distance to x_true')
plt.plot(d_pg_opt, label='PG: Distance to x_opt')
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.title(f'Proximal Gradient Method (p={p})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print(f"Proximal Gradient Method p={p}:")
print(f"最终迭代距离真值 = {d_pg_true[-1]:.4f}")
print(f"最终迭代距离最优解 = {d_pg_opt[-1]:.4f}")
