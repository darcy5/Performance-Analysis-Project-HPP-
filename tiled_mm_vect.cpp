#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <algorithm>

const int N = 3200;
const int TILE_L1 = 64;
const int TILE_L2 = 128;
const int TILE_L3 = 512;

using Matrix = std::vector<float>;  // Flattened matrix (row-major)

inline int idx(int i, int j) {
    return i * N + j;
}

void initialize_matrix(Matrix &mat, float value) {
    std::fill(mat.begin(), mat.end(), value);
}

void tiled_matrix_multiply(const Matrix &A, const Matrix &B, Matrix &C) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i3 = 0; i3 < N; i3 += TILE_L3) {
        for (int j3 = 0; j3 < N; j3 += TILE_L3) {
            for (int k3 = 0; k3 < N; k3 += TILE_L3) {

                for (int i2 = i3; i2 < std::min(i3 + TILE_L3, N); i2 += TILE_L2) {
                    for (int j2 = j3; j2 < std::min(j3 + TILE_L3, N); j2 += TILE_L2) {
                        for (int k2 = k3; k2 < std::min(k3 + TILE_L3, N); k2 += TILE_L2) {

                            for (int i1 = i2; i1 < std::min(i2 + TILE_L2, N); i1 += TILE_L1) {
                                for (int j1 = j2; j1 < std::min(j2 + TILE_L2, N); j1 += TILE_L1) {
                                    for (int k1 = k2; k1 < std::min(k2 + TILE_L2, N); k1 += TILE_L1) {

                                        for (int i = i1; i < std::min(i1 + TILE_L1, N); ++i) {
                                            for (int j = j1; j < std::min(j1 + TILE_L1, N); ++j) {
                                                float sum = C[idx(i, j)];

                                                //Vectorized innermost loop
                                                #pragma omp simd
                                                for (int k = k1; k < std::min(k1 + TILE_L1, N); ++k) {
                                                    sum += A[idx(i, k)] * B[idx(k, j)];
                                                }

                                                C[idx(i, j)] = sum;
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
}

int main() {
    Matrix A(N * N);
    Matrix B(N * N);
    Matrix C(N * N, 0.0f);

    initialize_matrix(A, 1.0f);
    initialize_matrix(B, 2.0f);

    auto start = std::chrono::high_resolution_clock::now();
    tiled_matrix_multiply(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Matrix multiplication completed in " << elapsed.count() << " seconds.\n";
    std::cout << "OpenMP threads used: " << omp_get_max_threads() << std::endl;

    return 0;
}
