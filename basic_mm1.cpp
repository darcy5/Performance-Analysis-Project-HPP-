#include <iostream>
#include <cstdlib>   
#include <ctime>       
#include <omp.h>       

#define M 2048
#define N 2048
#define K 2048
#define BLOCK_SIZE 64

#define fast_min(a, b) ((a) < (b) ? (a) : (b))

void initializeMatrix(float* mat, int size, bool zero = false) {
	#pragma omp parallel for
	for (int i = 0; i < size; ++i)
		mat[i] = zero ? 0.0f : std::rand() % 10;
}

// Matrix multiplication function using blocking + OpenMP + SIMD
void multiplyMatrices(const float* A, const float* B, float* C) {
	#pragma omp parallel for collapse(2)
	for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
		for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
			for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
				for (int i = ii; i < fast_min(ii + BLOCK_SIZE, M); ++i) {
					for (int j = jj; j < fast_min(jj + BLOCK_SIZE, N); ++j) {
						float sum = 0.0f;

						#pragma omp simd reduction(+:sum)
						for (int k = kk; k < fast_min(kk + BLOCK_SIZE, K); ++k) {
							sum += A[i * K + k] * B[k * N + j];
						}
						C[i * N + j] += sum;
					}
				}
			}
		}
	}
}

void cleanup(float* A, float* B, float* C) {
	std::free(A);
	std::free(B);
	std::free(C);
}

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
	initializeMatrix(C, M * N, true);  

	multiplyMatrices(A, B, C);

	std::cout << "Matrix multiplication completed.\n";

	cleanup(A, B, C);
	return 0;
}
