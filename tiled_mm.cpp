#include <iostream>
#include <vector>

#define N 2048
#define BLOCK_SIZE 64

void initializeMatrix(std::vector<int>& mat, int n) {
	for (int i = 0; i < n * n; ++i) {
		mat[i] = i % 10; 
	}
}

void multiplyMatrices(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, int n) {
	for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
		for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
			for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
				for (int i = ii; i < std::min(ii + BLOCK_SIZE, n); ++i) {
					for (int j = jj; j < std::min(jj + BLOCK_SIZE, n); ++j) {
						int sum = 0;
						for (int k = kk; k < std::min(kk + BLOCK_SIZE, n); ++k) {
							sum += A[i * n + k] * B[k * n + j];
						}
						C[i * n + j] += sum;
					}
				}
			}
		}
	}
}

int main() {
	std::vector<int> A(N * N), B(N * N), C(N * N, 0);

	initializeMatrix(A, N);
	initializeMatrix(B, N);

	std::cout << "Running single-threaded tiled matrix multiplication...\n";

	multiplyMatrices(A, B, C, N);

	std::cout << "Matrix multiplication completed.\n";
	return 0;
}
