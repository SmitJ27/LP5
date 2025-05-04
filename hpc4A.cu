/*

nvcc -o hpc4A hpc4A.cu
./hpc4A

Enter the size of vectors: 10
Enter elements of vector A:
1 2 3 4 5 6 7 8 9 10
Enter elements of vector B:
2 3 4 5 6 7 8 9 10 11
Result of vector addition:
3 5 7 9 11 13 15 17 19 21

*/

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel function for vector addition
__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    // Calculate the global index for each thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Perform addition if within bounds
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N;
    // Prompt user for the size of vectors
    cout << "Enter the size of vectors: "; 
    cin >> N;

    // Allocate memory for vectors A, B, and C on the host (CPU)
    int *h_A = new int[N], *h_B = new int[N], *h_C = new int[N];

    // Input elements for vector A
    cout << "Enter elements of vector A:\n";
    for (int i = 0; i < N; i++) {
        cin >> h_A[i];
    }

    // Input elements for vector B
    cout << "Enter elements of vector B:\n";
    for (int i = 0; i < N; i++) {
        cin >> h_B[i];
    }

    // Allocate memory for vectors A, B, and C on the device (GPU)
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(int));  // Memory for vector A on GPU
    cudaMalloc(&d_B, N * sizeof(int));  // Memory for vector B on GPU
    cudaMalloc(&d_C, N * sizeof(int));  // Memory for result vector C on GPU

    // Copy data from host to device (CPU -> GPU)
    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel to perform vector addition
    // (N + 255) / 256 is used to calculate the number of blocks needed
    // 256 is the number of threads per block
    vectorAdd<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);

    // Copy the result back from device to host (GPU -> CPU)
    cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the result of vector addition
    cout << "Result of vector addition:\n";
    for (int i = 0; i < N; i++) {
        cout << h_C[i] << " ";  // Print each element of the result
    }
    cout << endl;

    // Free memory on the device and host
    cudaFree(d_A);  // Free device memory for vector A
    cudaFree(d_B);  // Free device memory for vector B
    cudaFree(d_C);  // Free device memory for vector C
    delete[] h_A;   // Free host memory for vector A
    delete[] h_B;   // Free host memory for vector B
    delete[] h_C;   // Free host memory for vector C

    return 0;
}
