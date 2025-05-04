/*

nvcc -o hpc4B hpc4B.cu
./hpc4B

Enter the size of the square matrix: 2
Enter elements of matrix A:
1 2
3 4
Enter elements of matrix B:
5 6
7 8
Result of matrix multiplication:
19 22
43 50

*/

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel function for matrix multiplication
__global__ void matrixMul(int *A, int *B, int *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within bounds of the matrix
    if (row < n && col < n) {
        int sum = 0;
        // Perform the dot product of row and column
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    int N;
    cout << "Enter the size of the square matrix: "; 
    cin >> N;

    // Allocate memory for matrices A, B, and C on the host (CPU)
    int *h_A = new int[N * N], *h_B = new int[N * N], *h_C = new int[N * N];

    cout << "Enter elements of matrix A:\n";
    for (int i = 0; i < N * N; i++) {
        cin >> h_A[i];  // Input elements for matrix A
    }

    cout << "Enter elements of matrix B:\n";
    for (int i = 0; i < N * N; i++) {
        cin >> h_B[i];  // Input elements for matrix B
    }

    // Allocate memory for matrices A, B, and C on the device (GPU)
    int *d_A, *d_B, *d_C;
    int size = N * N * sizeof(int);

    cudaError_t err = cudaMalloc(&d_A, size);  // Allocate device memory for A
    if (err != cudaSuccess) {
        cerr << "CUDA malloc failed for A: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    err = cudaMalloc(&d_B, size);  // Allocate device memory for B
    if (err != cudaSuccess) {
        cerr << "CUDA malloc failed for B: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    err = cudaMalloc(&d_C, size);  // Allocate device memory for C
    if (err != cudaSuccess) {
        cerr << "CUDA malloc failed for C: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Set the number of threads per block (16x16 block)
    dim3 threadsPerBlock(16, 16);
    // Calculate the number of blocks per grid
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    // Launch the kernel to perform matrix multiplication
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    // Copy the result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Output the result of matrix multiplication
    cout << "Result of matrix multiplication:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << h_C[i * N + j] << " ";  // Print each element of the result matrix
        }
        cout << "\n";
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
