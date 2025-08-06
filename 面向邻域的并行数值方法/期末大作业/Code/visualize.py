import numpy as np
import matplotlib.pyplot as plt

def plot_sparsity(matrix_file="Matrix_n.dat"):
    try:
        with open(matrix_file) as f:
            n, nnz = map(int, f.readline().split())
            nA = list(map(float, f.readline().split()))
            JA = list(map(int, f.readline().split()))
            IA = list(map(int, f.readline().split()))
    except Exception as e:
        print(f"[×] Failed to read matrix: {e}")
        return

    row, col = [], []
    for i in range(n):
        for j in range(IA[i], IA[i + 1]):
            row.append(i)
            col.append(JA[j])

    plt.figure(figsize=(7, 7))
    plt.scatter(col, row, s=0.05, marker='.', color='black')
    plt.title("Sparsity Pattern of Matrix A")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("sparsity_pattern.png", dpi=300)
    plt.close()
    print("Sparsity pattern saved as: MatrixA_Structure.png")

def plot_temperature(data_file="Result_n.dat", Nx=1000, Ny=1000):
    try:
        temp = np.loadtxt(data_file)
        if temp.size != Nx * Ny:
            raise ValueError(f"Data size mismatch: {temp.size} != {Nx}×{Ny}")
        temp_2d = temp.reshape((Ny, Nx))

        plt.figure(figsize=(8, 6))
        im = plt.imshow(temp_2d, origin='lower', cmap='hot', aspect='auto')
        plt.colorbar(im, label='Temperature (K)')
        plt.title('2D Temperature Distribution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig("heatmap.png", dpi=300)
        plt.close()
        print("Heatmap saved as: heatmap.png")
    except Exception as e:
        print(f"[×] Failed to plot temperature: {e}")

if __name__ == "__main__":
    print("=== Visualization started ===")
    plot_sparsity()
    plot_temperature()
    print("=== Done ===")

