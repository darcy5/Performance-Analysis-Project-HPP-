#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>

std::mutex io_mutex;

const int N = 3200;        // Matrix size
const int TILE_SIZE = 128;  // Tile size
const int NUM_THREADS = std::thread::hardware_concurrency();

using Matrix = std::vector<std::vector<float>>;

// Initialization of the matrix with fixed values for reproducibility
void initialize_matrix(Matrix &mat, float value) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            mat[i][j] = value;
}

// Tiled matrix multiplication for a specific row range
void tiled_multiply_worker(const Matrix &A, const Matrix &B, Matrix &C, int row_start, int row_end) {
    {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cout << "Thread " << std::this_thread::get_id()
                  << " started: rows " << row_start << " to " << row_end - 1 << std::endl;
    }
    for (int i = row_start; i < row_end; i += TILE_SIZE) {
        for (int j = 0; j < N; j += TILE_SIZE) {
            for (int k = 0; k < N; k += TILE_SIZE) {
                for (int ii = i; ii < std::min(i + TILE_SIZE, row_end); ++ii) {
                    for (int jj = j; jj < std::min(j + TILE_SIZE, N); ++jj) {
                        float sum = C[ii][jj];
                        for (int kk = k; kk < std::min(k + TILE_SIZE, N); ++kk) {
                            sum += A[ii][kk] * B[kk][jj];
                        }
                        C[ii][jj] = sum;
                    }
                }
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cout << "Thread " << std::this_thread::get_id() << " finished.\n";
    }
}

//Multithreading
void tiled_matrix_multiply(const Matrix &A, const Matrix &B, Matrix &C) {
    std::vector<std::thread> threads;
    int rows_per_thread = N / NUM_THREADS; //distributes among the available threads

    for (int t = 0; t < NUM_THREADS; ++t) { //spawn threads
        int row_start = t * rows_per_thread;
        int row_end = (t == NUM_THREADS - 1) ? N : row_start + rows_per_thread;
        threads.emplace_back(tiled_multiply_worker, std::ref(A), std::ref(B), std::ref(C), row_start, row_end);
    }

    for (auto &thread : threads) //wait for all threads
        thread.join();
}

int main() {
    Matrix A(N, std::vector<float>(N));
    Matrix B(N, std::vector<float>(N));
    Matrix C(N, std::vector<float>(N, 0));

    initialize_matrix(A, 1.0f);
    initialize_matrix(B, 2.0f);

    auto start = std::chrono::high_resolution_clock::now();
    tiled_matrix_multiply(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Matrix multiplication completed in " << elapsed.count() << " seconds.\n";

    std::cout << "Detected hardware threads: " << NUM_THREADS << std::endl;

    return 0;
}
