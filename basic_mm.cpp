#include <iostream>
#include <vector>
#include <thread>
#include <random>

#define N 2048        
#define BLOCK_SIZE 64 
#define NUM_THREADS 4

void initializeMatrix(std::vector<int>& mat, int n) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(0, 9);

	for (int i = 0; i < n * n; ++i) {
		mat[i] = dist(gen);
	}
}

void multiplyMatrices(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C,
                      int n, int thread_id, int num_threads) {
	int rows_per_thread = n / num_threads;
	int row_start = thread_id * rows_per_thread;
	int row_end = (thread_id == num_threads - 1) ? n : row_start + rows_per_thread;

	for (int ii = row_start; ii < row_end; ii += BLOCK_SIZE) {
		for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
			for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
				for (int i = ii; i < std::min(ii + BLOCK_SIZE, row_end); ++i) {
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

	std::cout << "Running with " << NUM_THREADS << " threads...\n";

	std::vector<std::thread> threads;
	for (int t = 0; t < NUM_THREADS; ++t) {
		threads.emplace_back(multiplyMatrices, std::cref(A), std::cref(B), std::ref(C), N, t, NUM_THREADS);
	}
	for (auto& t : threads) {
		t.join();
	}

	std::cout << "Matrix multiplication completed.\n";
	return 0;
}
