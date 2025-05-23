#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <algorithm>

const int N = 4096;         // Matrix size
const int TILE = 128;        // Only one tile size used

using Matrix = std::vector<std::vector<float>>;

void initialize_matrix(Matrix &mat, float value) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            mat[i][j] = value;
}

void tiled_matrix_multiply(const Matrix &A, const Matrix &B, Matrix &C) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i0 = 0; i0 < N; i0 += TILE) {
        for (int j0 = 0; j0 < N; j0 += TILE) {
            for (int k0 = 0; k0 < N; k0 += TILE) {
                for (int i = i0; i < std::min(i0 + TILE, N); ++i) {
                    for (int j = j0; j < std::min(j0 + TILE, N); ++j) {
                        float sum = C[i][j];
                        for (int k = k0; k < std::min(k0 + TILE, N); ++k) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] = sum;
                    }
                }
            }
        }
    }
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
    std::cout << "OpenMP threads used: " << omp_get_max_threads() << std::endl;

    return 0;
}
