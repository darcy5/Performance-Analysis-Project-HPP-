#include <iostream>
#include <cstdlib>      // For rand(), malloc(), free()
#include <ctime>        // For seeding rand()
#include <omp.h>        // OpenMP

#define M 2048
#define N 2048
#define K 2048
#define BLOCK_SIZE 64

#define fast_min(a, b) ((a) < (b) ? (a) : (b))

// Initialize matrix with random values or zeros
void initializeMatrix(float* mat, int size, bool zero = false) {
	#pragma omp parallel for
	for (int i = 0; i < size; ++i)
		mat[i] = zero ? 0.0f : std::rand() % 10;
}

// Matrix multiplication using 3-level parallel blocking
void multiplyMatrices(const float* A, const float* B, float* C) {
	// First block of parallel loops: blockwise iteration
	#pragma omp parallel for collapse(2) schedule(dynamic)
	for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
		for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
			for (int kk = 0; kk < K; kk += BLOCK_SIZE) {

				// Second block: iterate over rows and cols within block
				for (int i = ii; i < fast_min(ii + BLOCK_SIZE, M); ++i) {
					for (int j = jj; j < fast_min(jj + BLOCK_SIZE, N); ++j) {
						float sum = 0.0f;

						// Third block: inner product within tile
						#pragma omp simd reduction(+:sum)
						for (int k = kk; k < fast_min(kk + BLOCK_SIZE, K); ++k) {
							sum += A[i * K + k] * B[k * N + j];
						}

						// Accumulate safely (required when C[i*N+j] updated by multiple threads)
						#pragma omp atomic
						C[i * N + j] += sum;
					}
				}
			}
		}
	}
}

// Free allocated matrices
void cleanup(float* A, float* B, float* C) {
	std::free(A);
	std::free(B);
	std::free(C);
}

// Main driver
int main() {
	std::srand(static_cast<unsigned>(std::time(0)));

	float* A = (float*) std::malloc(sizeof(float) * M * K);
	float* B = (float*) std::malloc(sizeof(float) * K * N);
	float* C = (float*) std::malloc(sizeof(float) * M * N);

	if (!A || !B || !C) {
		std::cerr << "Memory allocation failed!" << std::endl;
		return -1;
	}

	initializeMatrix(A, M * K);
	initializeMatrix(B, K * N);
	initializeMatrix(C, M * N, true);  // zero initialize C

	double start_time = omp_get_wtime();
	multiplyMatrices(A, B, C);
	double end_time = omp_get_wtime();

	std::cout << "Matrix multiplication completed in " 
	          << (end_time - start_time) << " seconds.\n";

	cleanup(A, B, C);
	return 0;
}
