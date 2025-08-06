// main_n.cpp  —— 并行 CG 求解 2-D 热传导线性方程组（根据 rel-res 判断解的精度）
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;

int n, nnz;
vector<double> nA;
vector<int>    JA;
vector<int>    IA;
vector<double> b;
vector<double> x;

void readMatrix(const string& filename) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error: cannot open " << filename << '\n';
        exit(1);
    }

    fin >> n >> nnz;
    nA.resize(nnz);
    JA.resize(nnz);
    IA.resize(n+1);
    b.resize(n);
    x.assign(n, 0.0);

    for (int i=0; i<nnz; ++i)
        fin >> nA[i];
    for (int i=0; i<nnz; ++i)
        fin >> JA[i];
    for (int i=0; i<=n; ++i)
        fin >> IA[i];
    for (int i=0; i<n; ++i)
        fin >> b[i];
}

void writeResult(const string& filename, const vector<double>& v) {
    ofstream fout(filename);
    if (!fout) {
        cerr << "Error: cannot write " << filename << '\n';
        exit(1);
    }
    for (double val : v)
    	fout << val << '\n';
}

void sparseMatVec(const vector<double>& vec, vector<double>& result) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j=IA[i]; j<IA[i+1]; ++j)
            sum += nA[j]*vec[JA[j]];
        result[i] = sum;
    }
}

double dotProduct(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i=0; i<a.size(); ++i)
        sum += a[i]*b[i];
    return sum;
}

double vectorNorm(const vector<double>& v) {
    return sqrt(dotProduct(v, v));
}

//返回 rel-res
double conjugateGradient(int maxIter = 5000, double tol = 1e-6) {
    vector<double> r(n), p(n), Ap(n);
    sparseMatVec(x, Ap);
    #pragma omp parallel for
    for (int i=0; i<n; ++i) 
        r[i]=b[i];
    p=r;

    double rsold = dotProduct(r, r);
    const double bNorm = sqrt(rsold);

    for (int k=0; k<maxIter; ++k) {
        sparseMatVec(p, Ap);
        double alpha = rsold/dotProduct(p, Ap);

        #pragma omp parallel for
        for (int i=0; i<n; ++i) {
            x[i] += alpha*p[i];
            r[i] -= alpha*Ap[i];
        }

        double rsnew = dotProduct(r, r);
        double rel_res = sqrt(rsnew) / bNorm;

        if (rel_res < tol) {
            cout << "[CG] converged: " << k + 1 << " iterations, rel-res = " << rel_res << '\n';
            return rel_res;
        }

        #pragma omp parallel for
        for (int i=0; i<n; ++i)
            p[i] = r[i] + (rsnew / rsold) * p[i];
        rsold = rsnew;
    }

    double final_rel = sqrt(rsold) / bNorm;
    cout << "[CG] reached maxIter = " << maxIter << ", rel-res = " << final_rel << '\n';
    return final_rel;
}

double computeResidual() {
    vector<double> Ax(n);
    sparseMatVec(x, Ax);
    #pragma omp parallel for
    for (int i=0; i<n; ++i)
        Ax[i] -= b[i];
    return vectorNorm(Ax);
}

void printMinMaxTemperature() {
    double minT = x[0], maxT = x[0];
#if _OPENMP >= 201511
    #pragma omp parallel for reduction(min:minT) reduction(max:maxT)
    for (int i=0; i<n; ++i) {
        minT = min(minT, x[i]);
        maxT = max(maxT, x[i]);
    }
#else
    #pragma omp parallel
    {
        double tmin = minT, tmax = maxT;
        #pragma omp for nowait
        for (int i=0; i<n; ++i) {
            tmin = min(tmin, x[i]);
            tmax = max(tmax, x[i]);
        }
        #pragma omp critical
        {
            minT = min(minT, tmin);
            maxT = max(maxT, tmax);
        }
    }
#endif
    cout << "Temperature range = [" << minT << ", " << maxT << "]\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <num_threads> <matrix_file>\n";
        return 1;
    }

    int numThreads = atoi(argv[1]);
    string matrixFile = argv[2];
    omp_set_num_threads(numThreads);

    cout << ">>> Using OpenMP with " << numThreads << " threads\n";
    readMatrix(matrixFile);
    cout << ">>> Matrix: n = " << n << ", nnz = " << nnz << '\n';

    cout << "Initial residual = " << vectorNorm(b) << '\n';

    auto t0 = chrono::high_resolution_clock::now();
    double rel_res = conjugateGradient();  // 捕获返回的 rel-res
    auto t1 = chrono::high_resolution_clock::now();
    double residual = computeResidual();

    chrono::duration<double> dt = t1 - t0;
    writeResult("Result_n.dat", x);

    cout << "Residual norm  = " << residual << '\n'
         << "Elapsed time   = " << dt.count() << " s\n";
    printMinMaxTemperature();

    cout << (rel_res <= 1e-6 ? "Solution is numerically correct.\n"
                             : "Solution is NOT accurate enough.\n");
    return 0;
}

