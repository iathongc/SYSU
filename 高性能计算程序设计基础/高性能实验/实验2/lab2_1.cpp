#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

#define ARRAY_SIZE 1000
#define MAX_ELEMENTS_PER_THREAD 10

int a[ARRAY_SIZE];
int global_index = 0;
int total_sum = 0;
pthread_mutex_t mutex;

void* sum_array(void* threadid) {
    int local_sum = 0;
    while (1) {
        int start_index;
        int count = 0;

        pthread_mutex_lock(&mutex);
        start_index = global_index;
        if (global_index < ARRAY_SIZE) {
            count = (ARRAY_SIZE - global_index < MAX_ELEMENTS_PER_THREAD) ? (ARRAY_SIZE - global_index) : MAX_ELEMENTS_PER_THREAD;
            global_index += count;
        }
        pthread_mutex_unlock(&mutex);

        if (count == 0) {
            break;
        }

        for (int i = 0; i < count; i++) {
            local_sum += a[start_index + i];
        }
    }

    pthread_mutex_lock(&mutex);
    total_sum += local_sum;
    pthread_mutex_unlock(&mutex);

    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <number_of_threads>\n", argv[0]);
        exit(-1);
    }

    int num_threads = atoi(argv[1]);
    if (num_threads <= 0) {
        printf("Number of threads should be greater than 0.\n");
        exit(-1);
    }

    pthread_t threads[num_threads];
    pthread_mutex_init(&mutex, NULL);

    // Initialize array with i + 1
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = i + 1;
    }

    // Start timing
    clock_t start_time = clock();

    // Create threads
    for (long t = 0; t < num_threads; t++) {
        int rc = pthread_create(&threads[t], NULL, sum_array, (void*)t);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    // Join threads
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    // Stop timing
    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Total sum = %d\n", total_sum);
    printf("Execution time = %f seconds\n", elapsed_time);

    pthread_mutex_destroy(&mutex);
    pthread_exit(NULL);
}

