#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
using namespace std;

int n, nnz;
vector<double> nA;  // 非零元素值
vector<int> JA;     // 非零元素的列索引
vector<int> IA;     // 每行起始索引
vector<double> b;   // 右端项（为0）
vector<double> x;   // 解向量

void readMatrix(const string& filename) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error: Cannot open file " << filename << endl;
        exit(1);
    }

    fin >> n >> nnz;
    nA.resize(nnz);
    JA.resize(nnz);
    IA.resize(n + 1);
    b.resize(n, 0.0);  // 全部为0
    x.resize(n, 0.0);  // 初始解为0

    for (int i = 0; i < nnz; ++i)
        fin >> nA[i];
    for (int i = 0; i < nnz; ++i)
        fin >> JA[i];
    for (int i = 0; i <= n; ++i)
        fin >> IA[i];
    for (int i = 0; i < n; ++i)
        fin >> b[i];

    fin.close();
}

void writeResult(const string& filename, const vector<double>& result) {
    ofstream fout(filename);
    if (!fout) {
        cerr << "Error: Cannot write to " << filename << endl;
        exit(1);
    }
    for (int i=0; i<result.size(); ++i)
        fout << result[i] << endl;
    fout.close();
}

void sparseMatVec(const vector<double>& vec, vector<double>& result) {
    #pragma omp parallel for
    for (int i=0; i<n; ++i) {
        double sum = 0.0;
        for (int j=IA[i]; j<IA[i+1]; ++j) {
            sum += nA[j]*vec[JA[j]];
        }
        result[i] = sum;
    }
}

double dotProduct(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i=0; i<a.size(); ++i)
        sum += a[i]*b[i];
    return sum;
}

double vectorNorm(const vector<double>& v) {
    return sqrt(dotProduct(v, v));
}

void conjugateGradient(int maxIter=10000, double tol=1e-6) {
    vector<double> r(n), p(n), Ap(n);
    vector<double> Ax(n);
    sparseMatVec(x, Ax);
    #pragma omp parallel for
    for (int i=0; i<n; ++i)
        r[i] = b[i]-Ax[i];
    p = r;

    double rsold = dotProduct(r, r);
    for (int k=0; k<maxIter; ++k) {
        sparseMatVec(p, Ap);
        double alpha = rsold / dotProduct(p, Ap);

        #pragma omp parallel for
        for (int i=0; i<n; ++i)
            x[i] += alpha*p[i];

        #pragma omp parallel for
        for (int i=0; i<n; ++i)
            r[i] -= alpha*Ap[i];

        double rsnew = dotProduct(r, r);
        if (sqrt(rsnew) < tol)
            break;

        #pragma omp parallel for
        for (int i=0; i<n; ++i)
            p[i] = r[i] + (rsnew/rsold) * p[i];

        rsold = rsnew;
    }
}

double computeResidual() {
    vector<double> Ax(n);
    sparseMatVec(x, Ax);
    vector<double> r(n);
    #pragma omp parallel for
    for (int i=0; i<n; ++i)
        r[i] = Ax[i]-b[i];
    return vectorNorm(r);
}

void printMatrixInfo() {
    cout << "Matrix dimension n = " << n << ", non-zeros nnz = " << nnz << endl;

    cout << "\nFirst 3 elements of nA: ";
    for (int i=0; i<min(3, (int)nA.size()); ++i)
        cout << nA[i] << " ";
    cout << "\nFirst 3 elements of JA: ";
    for (int i=0; i<min(3, (int)JA.size()); ++i)
        cout << JA[i] << " ";
    cout << "\nFirst 3 elements of IA: ";
    for (int i=0; i<min(3, (int)IA.size()); ++i)
        cout << IA[i] << " ";

    cout << "\n\nFirst 3 elements of b: ";
    for (int i=0; i<min(3, (int)b.size()); ++i)
        cout << b[i] << " ";
    cout << "\n" << endl;
}

int main(int argc, char* argv[]) {
    if (argc<2) {
        cout << "Usage: ./solver <num_threads>" << endl;
        return 1;
    }

    int num_threads = atoi(argv[1]);
    omp_set_num_threads(num_threads);
    cout << "Using OpenMP with " << num_threads << " threads." << endl;

    readMatrix("Matrix.dat");

    printMatrixInfo(); 

    auto start = chrono::high_resolution_clock::now();
    conjugateGradient();
    auto end = chrono::high_resolution_clock::now();

    writeResult("Result.dat", x);

    double residual = computeResidual();
    cout << "Residual norm: " << residual << endl;

    chrono::duration<double> elapsed = end - start;
    cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;

    if (residual <= 1e-6)
        cout << "Solution is numerically correct." << endl;
    else
        cout << "Solution is NOT accurate enough." << endl;

    return 0;
}

