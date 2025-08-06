// generate_matrix_regular.cpp
// 构造 1000x1000 网格上的二维稳态热传导问题稀疏矩阵（CSR 格式）
// 左边恒温 300K，右边恒温 1000K；上下为 Neumann 边界（绝热）
// 输出文件 Matrix_n.dat：包含 n、nnz、nA、JA、IA、b

#include <fstream>
#include <vector>
#include <iostream>

int main() {
    const int Nx = 1000, Ny = 1000;
    const int N  = Nx * Ny;
    const double T_left  = 300.0;
    const double T_right = 1000.0;

    std::vector<double> nA;
    std::vector<int>    JA;
    std::vector<int>    IA(N + 1, 0);
    std::vector<double> b(N, 0.0);

    nA.reserve(5ull * N); JA.reserve(5ull * N);

    for (int j=0, k=0; j<Ny; ++j) {
        for (int i=0; i<Nx; ++i, ++k) {
            IA[k] = static_cast<int>(nA.size());

            bool left   = (i == 0),        right = (i == Nx - 1);
            bool bottom = (j == 0),        top   = (j == Ny - 1);

            if (left) {
                nA.push_back(1.0); JA.push_back(k);
                b[k] = T_left;
                continue;
            }

            if (right) {
                nA.push_back(1.0); JA.push_back(k);
                b[k] = T_right;
                continue;
            }

            // 内部或 Neumann 边界点
            nA.push_back(4.0);  JA.push_back(k);  // 对角项先设为 4.0

            // 左邻
            if (i>1) {
                nA.push_back(-1.0); JA.push_back(k - 1);
            }
            else {
                b[k] += T_left;
            }

            // 右邻
            if (i < Nx-2) {
                nA.push_back(-1.0); JA.push_back(k + 1);
            }
            else {
                b[k] += T_right;
            }

            // 上 / 下边界的 Neumann 条件（∂T/∂n = 0）
            bool isNeumannY = false;

            if (!bottom) {
                nA.push_back(-1.0); JA.push_back(k - Nx);
            }
            else {
                isNeumannY = true;
            }

            if (!top) {
                nA.push_back(-1.0); JA.push_back(k + Nx);
            }
            else {
                isNeumannY = true;
            }

            if (isNeumannY) {
                nA[IA[k]] = 3.0;  // 把对角项从 4 改为 3
            }
        }
    }

    IA[N] = static_cast<int>(nA.size());

    std::ofstream fout("Matrix_n.dat");
    if (!fout) {
        std::cerr << "Error: Cannot open Matrix_n.dat for writing.\n";
        return 1;
    }

    fout << N << ' ' << nA.size() << '\n';
    for (double v : nA) fout << v << ' ';
    fout << '\n';
    for (int c : JA)   fout << c << ' ';
    fout << '\n';
    for (int r : IA)   fout << r << ' ';
    fout << '\n';
    for (double v : b) fout << v << '\n';

    std::cout << "Matrix_n.dat generated: N=" << N
              << "  nnz=" << nA.size() << '\n' << std::endl;
    return 0;
}

