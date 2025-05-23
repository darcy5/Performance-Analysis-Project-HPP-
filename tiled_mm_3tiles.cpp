#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>
#include <algorithm>

std::mutex io_mutex;

const int N = 3200;        // Matrix size
const int TILE_L1 = 64;    // Fits in L1 cache of size 160 KB
const int TILE_L2 = 128;   // Fits in L2 cache of size 2.5 MB
const int TILE_L3 = 512;   // Fits in L3 cache of size 6.0 MB
const int NUM_THREADS = std::thread::hardware_concurrency();

using Matrix = std::vector<std::vector<float>>;

// Initialize matrices with a constant value
void initialize_matrix(Matrix &mat, float value) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            mat[i][j] = value;
}

// 3-level tiled matrix multiplication per thread
void tiled_multiply_worker(const Matrix &A, const Matrix &B, Matrix &C, int row_start, int row_end) {
    {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cout << "Thread " << std::this_thread::get_id()
                  << " started: rows " << row_start << " to " << row_end - 1 << std::endl;
    }

    for (int i3 = row_start; i3 < row_end; i3 += TILE_L3) {
        for (int j3 = 0; j3 < N; j3 += TILE_L3) {
            for (int k3 = 0; k3 < N; k3 += TILE_L3) {

                for (int i2 = i3; i2 < std::min(i3 + TILE_L3, row_end); i2 += TILE_L2) {
                    for (int j2 = j3; j2 < std::min(j3 + TILE_L3, N); j2 += TILE_L2) {
                        for (int k2 = k3; k2 < std::min(k3 + TILE_L3, N); k2 += TILE_L2) {

                            for (int i1 = i2; i1 < std::min(i2 + TILE_L2, row_end); i1 += TILE_L1) {
                                for (int j1 = j2; j1 < std::min(j2 + TILE_L2, N); j1 += TILE_L1) {
                                    for (int k1 = k2; k1 < std::min(k2 + TILE_L2, N); k1 += TILE_L1) {

                                        for (int i = i1; i < std::min(i1 + TILE_L1, row_end); ++i) {
                                            for (int j = j1; j < std::min(j1 + TILE_L1, N); ++j) {
                                                float sum = C[i][j];
                                                for (int k = k1; k < std::min(k1 + TILE_L1, N); ++k) {
                                                    sum += A[i][k] * B[k][j];
                                                }
                                                C[i][j] = sum;
                                            }
                                        }

                                    }
                                }
                            }

                        }
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

// Spawns threads
void tiled_matrix_multiply(const Matrix &A, const Matrix &B, Matrix &C) {
    std::vector<std::thread> threads;
    int rows_per_thread = N / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; ++t) {
        int row_start = t * rows_per_thread;
        int row_end = (t == NUM_THREADS - 1) ? N : row_start + rows_per_thread;
        threads.emplace_back(tiled_multiply_worker, std::ref(A), std::ref(B), std::ref(C), row_start, row_end);
    }

    for (auto &thread : threads)
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
