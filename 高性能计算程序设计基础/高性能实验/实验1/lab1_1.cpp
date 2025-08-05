#include <iostream>
#include <random>
#include <mpi.h>

using namespace std;

const int MATRIX_SIZE = 2048; //Decide whether to print the result
const int MASTER_PROCESS = 0;

void initMatrix(double* mat, int numElements, unsigned int seed) {
    mt19937 generator(seed);
    uniform_real_distribution<double> distribution(0.0, 1.0);
    
    for (int i = 0; i < numElements; ++i) {
        mat[i] = distribution(generator);
    }
}

void multiplyMatrices(const double* matA, const double* matB, double* matC, int dimension) {
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            double sum = 0.0;
            for (int k = 0; k < dimension; ++k) {
                sum += matA[i * dimension + k] * matB[k * dimension + j];
            }
            matC[i * dimension + j] = sum;
        }
    }
}

void showMatrix(const double* mat, int fullSize) {
    int partSize = 2;  // Show the top-left 2x2 submatrix
    for (int i = 0; i < partSize; ++i) {
        for (int j = 0; j < partSize; ++j) {
            cout << mat[i * fullSize + j] << " ";
        }
        cout << "... ";
        cout << endl;
    }
    cout << "... " << endl;
}

int main(int argc, char* argv[]) {
    int processRank, numProcesses;
    double *matA = nullptr, *matB = nullptr, *matC = nullptr;
    double start, end;
    unsigned int seed1 = 12345; 
    unsigned int seed2 = 54321; 

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    
    int showOutput = 0;
    int localMatSize = (numProcesses > 1) ? MATRIX_SIZE / (numProcesses - 1) : MATRIX_SIZE;

    if (processRank == MASTER_PROCESS) {
        matA = new double[MATRIX_SIZE * MATRIX_SIZE];
        matB = new double[MATRIX_SIZE * MATRIX_SIZE];
        matC = new double[MATRIX_SIZE * MATRIX_SIZE];

        initMatrix(matA, MATRIX_SIZE * MATRIX_SIZE, seed1);
        initMatrix(matB, MATRIX_SIZE * MATRIX_SIZE, seed2);
    }
    else {
        matA = new double[MATRIX_SIZE * localMatSize];
        matB = new double[MATRIX_SIZE * MATRIX_SIZE];
        matC = new double[MATRIX_SIZE * localMatSize];
    }
    
    start = MPI_Wtime();
    
    MPI_Bcast(matB, MATRIX_SIZE * MATRIX_SIZE, MPI_DOUBLE, MASTER_PROCESS, MPI_COMM_WORLD);
    
    if (numProcesses > 1) {
        if (processRank == MASTER_PROCESS) {
            for (int dest = 1; dest < numProcesses; ++dest) {
                int startIdx = (dest - 1) * localMatSize;
                MPI_Send(&matA[startIdx * MATRIX_SIZE], localMatSize * MATRIX_SIZE, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }
        }
        else {
            MPI_Recv(matA, localMatSize * MATRIX_SIZE, MPI_DOUBLE, MASTER_PROCESS, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            multiplyMatrices(matA, matB, matC, localMatSize);
            MPI_Send(matC, localMatSize * MATRIX_SIZE, MPI_DOUBLE, MASTER_PROCESS, 0, MPI_COMM_WORLD);
        }
    }
    else {
        multiplyMatrices(matA, matB, matC, MATRIX_SIZE);
    }
    
    if (processRank == MASTER_PROCESS) {
        for (int source = 1; source < numProcesses; ++source) {
            int startIdx = (source - 1) * localMatSize;
            MPI_Recv(&matC[startIdx * MATRIX_SIZE], localMatSize * MATRIX_SIZE, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (showOutput) {
            cout << "Matrix A: " << endl;
            showMatrix(matA, MATRIX_SIZE);
			cout << endl;
            cout << "Matrix B:" << endl;
            showMatrix(matB, MATRIX_SIZE);
			cout << endl;
            cout << "Matrix C: " << endl;
            showMatrix(matC, MATRIX_SIZE);
        }

        end = MPI_Wtime();
        cout << "Execution time: " << (end - start) * 1000 << " ms" << endl;

        delete[] matA;
        delete[] matB;
        delete[] matC;
    }
    else {
        delete[] matA;
        delete[] matB;
        delete[] matC;
    }
    MPI_Finalize();
    return 0;
}
