#include <iostream>
#include <random>
#include <mpi.h>

void generateRandomMatrix(double arr[], int length, unsigned int seed) {
    std::mt19937 gen(seed);  // Mersenne Twister 19937 generator
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = 0;i < length; ++i) {
        arr[i] = distribution(gen);  // Fill the matrix with random values
    }
}

void multiplyBlock(double* matrixA, double* matrixB, double* matrixC, int dim, int blockRows) {
    for (int row = 0; row < blockRows; ++row) {
        for (int col = 0; col < dim; ++col) {
            matrixC[row * dim + col] = 0.0;
            for (int inner = 0; inner < dim; ++inner) {
                matrixC[row * dim + col] += matrixA[row * dim + inner] * matrixB[inner * dim + col];
            }
        }
    }
}

void displayMatrix(double* mat, int dim, int rowsToPrint, int colsToPrint) {
    for (int i = 0; i < rowsToPrint; ++i) {
        for (int j = 0; j < colsToPrint; ++j) {
            std::cout << mat[i * dim + j] << " ";
        }
        std::cout << "...";  // Omit remaining columns for brevity
        std::cout << std::endl;
    }
    if (rowsToPrint < dim) {
        std::cout << "...";  // Omit remaining rows for brevity
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int processRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    
    const int matrixDim = 2048; // Global matrix dimension
    const int rowsPerProc = matrixDim / worldSize; // Rows assigned to each process
    const int showRows = 2; // Number of rows to display
    const int showCols = 2; // Number of columns to display
    bool printResults = false; //Decide whether to print the result

    double* fullMatrixA = new double[matrixDim * matrixDim];  // Full matrix A
    double* fullMatrixB = new double[matrixDim * matrixDim];  // Full matrix B
    double* resultMatrix = new double[matrixDim * matrixDim]; // Result matrix C
    double* localBlockA = new double[rowsPerProc * matrixDim]; // Local block of A
    double* localBlockC = new double[rowsPerProc * matrixDim]; // Local block of C

    // Initialize matrices A and B in process 0
    if (processRank == 0) {
        generateRandomMatrix(fullMatrixA, matrixDim * matrixDim, 42);
        generateRandomMatrix(fullMatrixB, matrixDim * matrixDim, 24);
    }
    
    double startTime = MPI_Wtime(); // Start timing

    // Scatter matrix A to all processes
    MPI_Scatter(fullMatrixA, rowsPerProc * matrixDim, MPI_DOUBLE, localBlockA, rowsPerProc * matrixDim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Broadcast matrix B to all processes
    MPI_Bcast(fullMatrixB, matrixDim * matrixDim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform local block multiplication
    multiplyBlock(localBlockA, fullMatrixB, localBlockC, matrixDim, rowsPerProc);
    
    // Gather results from all processes
    MPI_Gather(localBlockC, rowsPerProc * matrixDim, MPI_DOUBLE, resultMatrix, rowsPerProc * matrixDim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double endTime = MPI_Wtime(); // End timing

    // Output results in process 0
    if (processRank == 0) {
        if (printResults) {
            std::cout << "Matrix A:" << std::endl;
            displayMatrix(fullMatrixA, matrixDim, showRows, showCols);

            std::cout << "\nMatrix B:" << std::endl;
            displayMatrix(fullMatrixB, matrixDim, showRows, showCols);

            std::cout << "\nMatrix C:" << std::endl;
            displayMatrix(resultMatrix, matrixDim, showRows, showCols);
        }
        std::cout << "Execution time: " << (endTime - startTime) * 1000 << " ms" << std::endl;
    }

    // Clean up memory
    delete[] fullMatrixA;
    delete[] fullMatrixB;
    delete[] resultMatrix;
    delete[] localBlockA;
    delete[] localBlockC;

    MPI_Finalize();

    return 0;
}
