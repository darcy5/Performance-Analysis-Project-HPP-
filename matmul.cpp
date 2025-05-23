#include <iostream>
#include <omp.h>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

#define N 2048         // Matrix dimension
//#define BLOCK_SIZE 128 // Loop tiling block size
#define BLOCK_SIZE 64

// Thread-safe random number initialization
void initializeMatrix(std::vector<float>& mat, int size) {
	#pragma omp parallel
	{
		std::mt19937 rng;
		rng.seed(std::random_device{}() + omp_get_thread_num());
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);

		#pragma omp for
		for (int i = 0; i < size; ++i) {
			mat[i] = dist(rng);
		}
	}
}

// Transpose matrix B to improve cache performance
void transposeMatrix(const std::vector<float>& src, std::vector<float>& dst, int n) {
	#pragma omp parallel for collapse(2) schedule(dynamic)
	//#pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			dst[j * n + i] = src[i * n + j];
		}
	}
}

// Optimized matrix multiplication using tiling, transposed B, and OpenMP
void multiplyMatrices(const std::vector<float>& A, const std::vector<float>& B_T, std::vector<float>& C, int n) {
	#pragma omp parallel for collapse(3) schedule(dynamic)
	//#pragma omp parallel for collapse(2) schedule(static, 2)
	for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
		for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
			for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
				for (int i = ii; i < std::min(ii + BLOCK_SIZE, n); ++i) {
					for (int k = kk; k < std::min(kk + BLOCK_SIZE, n); ++k) {
						float a = A[i * n + k];
						for (int j = jj; j < std::min(jj + BLOCK_SIZE, n); ++j) {
							C[i * n + j] += a * B_T[j * n + k]; // Access B_T row-wise
						}
					}
				}
			}
		}
	}
}

int main() {
	std::vector<float> A(N * N), B(N * N), B_T(N * N), C(N * N, 0.0f);

	initializeMatrix(A, N * N);
	initializeMatrix(B, N * N);

	// Transpose B for better access during multiplication
	transposeMatrix(B, B_T, N);

	int num_threads = 4;  // For 4-core CPU
	omp_set_num_threads(num_threads);
	std::cout << "Running with " << num_threads << " threads.\n";

	double start = omp_get_wtime();
	multiplyMatrices(A, B_T, C, N);
	double end = omp_get_wtime();

	std::cout << "Multiplication completed in " << (end - start) << " seconds.\n";
	return 0;
}
