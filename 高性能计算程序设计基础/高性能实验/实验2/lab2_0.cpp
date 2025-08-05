#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <iostream>
#include <stdint.h>

int rowsA, colsA, colsB;
double **matA;
double **matB;
double **matC;
int num_threads;
bool print_matrices = true; 

void initialize_matrices(int m, int n, int k) {
    srand(243);
    matA = new double*[m];
    matB = new double*[n];
    matC = new double*[m];
    for (int i = 0; i < m; i++) {
        matA[i] = new double[n];
        for (int j = 0; j < n; j++) {
            matA[i][j] = static_cast<double>(rand() % 1000 / 10.0);
        }
    }
    for (int i = 0; i < n; i++) {
        matB[i] = new double[k];
        for (int j = 0; j < k; j++) {
            matB[i][j] = static_cast<double>(rand() % 1000 / 10.0);
        }
    }
    for (int i = 0; i < m; i++) {
        matC[i] = new double[k];
        for (int j = 0; j < k; j++) {
            matC[i][j] = 0.0;
        }
    }
}

void display_matrix(int row, int col, double **matrix) {
    int print_rows = std::min(row, 2);
    int print_cols = std::min(col, 2);

    for (int i = 0; i < print_rows; i++) {
        for (int j = 0; j < print_cols; j++) {
            std::cout << matrix[i][j] << "\t";
        }
        if (col > 2) {
            std::cout << "...\t";
        }
        std::cout << std::endl;
    }

    if (row > 2) {
        for (int j = 0; j < print_cols; j++) {
            std::cout << "...\t";
        }
        if (col > 2) {
            std::cout << "...\t";
        }
        std::cout << std::endl;
    }
}

void *matrix_multiplication(void *thread_id) {
    intptr_t tid = (intptr_t)thread_id;
    int start_row, end_row;
    int base_rows = rowsA / num_threads;
    int extra_rows = rowsA % num_threads;
    int assigned_rows = 0;

    if (tid < extra_rows) {
        assigned_rows = base_rows + 1;
        start_row = tid * assigned_rows;
    }
     else {
        assigned_rows = base_rows;
        start_row = tid * assigned_rows + extra_rows;
    }
    end_row = start_row + assigned_rows;

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < colsB; j++) {
            matC[i][j] = 0.0;
            for (int k = 0; k < colsA; k++) {
                matC[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <rowsA> <colsA> <colsB> <num_threads> [print_matrices]" << std::endl;
        return 1;
    }

    rowsA = atoi(argv[1]);
    colsA = atoi(argv[2]);
    colsB = atoi(argv[3]);
    num_threads = atoi(argv[4]);

    if (argc > 5) {
        print_matrices = (atoi(argv[5]) != 0); 
    }

    initialize_matrices(rowsA, colsA, colsB);

    if (print_matrices) {
        std::cout << "Matrix A:" << std::endl;
        display_matrix(rowsA, colsA, matA);
        std::cout << std::endl;
        std::cout << "Matrix B:" << std::endl;
        display_matrix(colsA, colsB, matB);
    }

    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

    clock_t start = clock();
    for (intptr_t t = 0; t < num_threads; t++) {
        pthread_create(&threads[t], NULL, matrix_multiplication, (void *)t);
    }
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    clock_t finish = clock();

    double elapsed_time = static_cast<double>(finish - start) / CLOCKS_PER_SEC;

    if (print_matrices) {
        std::cout << std::endl;
        std::cout << "Result Matrix C:" << std::endl;
        display_matrix(rowsA, colsB, matC);
        std::cout << std::endl;
    }

    std::cout << "Execution Time: " << elapsed_time << " s" << std::endl;

    free(threads);

    // Free allocated memory for matrices
    for (int i = 0; i < rowsA; i++) {
        delete[] matA[i];
        delete[] matC[i];
    }
    delete[] matA;
    delete[] matC;

    for (int i = 0; i < colsA; i++) {
        delete[] matB[i];
    }
    delete[] matB;

    return 0;
}

