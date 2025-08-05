#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void printMatrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i < 2 && j < 2) {
                printf("%5.2f ", matrix[i * cols + j]);
            } else if (i < 2 && j == 2) {
                printf("... ");
                break;
            }
        }
        if (i == 2) {
            printf("...\n");
            break;
        } else {
            printf("\n");
        }
    }
}

int main(int argc, char **argv) {
    int rank, size, rows, cols;
    double *A, *B, *C;
    double startTime, endTime;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter number of rows and columns: ");
        scanf("%d %d", &rows, &cols);

        // Allocate memory for matrices
        A = (double *)malloc(rows * cols * sizeof(double));
        B = (double *)malloc(cols * cols * sizeof(double));  // Assuming square B for simplicity
        C = (double *)malloc(rows * cols * sizeof(double));

        // Initialize matrices A and B with random numbers
        srand(time(NULL));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                A[i * cols + j] = rand() % 10;
                B[i * cols + j] = rand() % 10;
            }
        }
    }

    // Broadcast dimensions to all processes
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Distribute matrix data to other processes
    // Assuming a simple scenario for this example
    if (rank == 0) {
        startTime = MPI_Wtime();

        // Perform matrix multiplication A * B = C
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                C[i * cols + j] = 0;
                for (int k = 0; k < cols; k++) {
                    C[i * cols + j] += A[i * cols + k] * B[k * cols + j];
                }
            }
        }

        endTime = MPI_Wtime();

        // Print first 2x2 sub-matrices with ellipsis
        printf("\n");
        printf("Matrix A (first 2x2 with ellipsis):\n");
        printMatrix(A, rows, cols);
	
	printf("\n");
        printf("Matrix B (first 2x2 with ellipsis):\n");
        printMatrix(B, cols, cols);

	printf("\n");
        printf("Matrix C (first 2x2 with ellipsis):\n");
        printMatrix(C, rows, cols);
	
	printf("\n");
        printf("Execution time: %f seconds\n", endTime - startTime);

        // Free memory
        free(A);
        free(B);
        free(C);
    }

    MPI_Finalize();
    return 0;
}

