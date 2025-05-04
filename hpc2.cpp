/*

Step 1 - (if its already installed, go to step 2)
sudo apt-get update
sudo apt-get install g++

Step 2 - 
g++ -fopenmp hpc2.cpp -o hpc2

Step 3 - 
./hpc2

sample input - 

Enter number of elements: 6
Enter elements: 5 3 8 1 2 7

sample output - 

Sequential Bubble Sort: 1 2 3 5 7 8 
Time: 0.001234 sec

Parallel Bubble Sort: 1 2 3 5 7 8 
Time: 0.000876 sec

Sequential Merge Sort: 1 2 3 5 7 8 
Time: 0.001456 sec

Parallel Merge Sort: 1 2 3 5 7 8 
Time: 0.000654 sec

*/


#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>  // For setting precision
using namespace std;

// 1. Sequential Bubble Sort
void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    // Outer loop for number of passes
    for (int i = 0; i < n - 1; i++) {
        // Inner loop to compare adjacent elements and swap if necessary
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);  // Swap adjacent elements
            }
        }
    }
}

// 2. Sequential Merge Sort
// Merges two halves of the array into a sorted array
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);  // Temporary array to hold merged elements
    int i = left, j = mid + 1, k = 0;
    // Merging two sorted halves
    while (i <= mid && j <= right) {
        temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];  // Take the smaller element
    }
    // Copy the remaining elements from the left half
    while (i <= mid) {
        temp[k++] = arr[i++];
    }
    // Copy the remaining elements from the right half
    while (j <= right) {
        temp[k++] = arr[j++];
    }
    // Copy the sorted elements back into the original array
    for (int m = 0; m < k; m++) {
        arr[left + m] = temp[m];
    }
}

// Recursive function to implement merge sort
void sequentialMergeSort(vector<int>& arr, int left, int right) {
    if (left >= right) {
        return;  // Base case: if the array has 1 or 0 elements, it is already sorted
    }
    int mid = left + (right - left) / 2;  // Find the middle point
    sequentialMergeSort(arr, left, mid);  // Recursively sort the left half
    sequentialMergeSort(arr, mid + 1, right);  // Recursively sort the right half
    merge(arr, left, mid, right);  // Merge the sorted halves
}

// 3. Parallel Bubble Sort (Odd-Even Phase)
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    bool sorted = false;
    while (!sorted) {
        sorted = true;
        // Parallel phase 1: Compare and swap odd-indexed pairs
        #pragma omp parallel for shared(arr, sorted)
        for (int i = 1; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                sorted = false;
            }
        }
        // Parallel phase 2: Compare and swap even-indexed pairs
        #pragma omp parallel for shared(arr, sorted)
        for (int i = 0; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                sorted = false;
            }
        }
    }
}

// 4. Parallel Merge Sort
// Recursive function to implement parallel merge sort
void parallelMergeSort(vector<int>& arr, int left, int right) {
    if (left >= right) {
        return;  // Base case: if the array has 1 or 0 elements, it is already sorted
    }
    int mid = left + (right - left) / 2;  // Find the middle point
    // Use OpenMP parallel sections to divide the work
    #pragma omp parallel sections
    {
        // Recursively sort the left half in one parallel section
        #pragma omp section
        parallelMergeSort(arr, left, mid);
        // Recursively sort the right half in another parallel section
        #pragma omp section
        parallelMergeSort(arr, mid + 1, right);
    }
    merge(arr, left, mid, right);  // Merge the sorted halves
}

int main() {
    int n;
    cout << "Enter number of elements: "; 
    cin >> n;  // Input the number of elements
    vector<int> arr(n), arr2;  // Create vectors to store original and copied arrays
    cout << "Enter elements: ";
    for (int& x : arr) {
        cin >> x;  // Input elements into the array
    }

    // Sequential Bubble Sort
    arr2 = arr;  // Copy original array to arr2 for sorting
    double start = omp_get_wtime();  // Start the timer
    sequentialBubbleSort(arr2);  // Perform sequential bubble sort
    double end = omp_get_wtime();  // End the timer
    cout << "Sequential Bubble Sort: ";
    for (int x : arr2) {
        cout << x << " ";  // Output the sorted array
    }
    cout << "\nTime: " << fixed << setprecision(6) << (end - start) << " sec\n";  // Display the time taken

    // Parallel Bubble Sort
    arr2 = arr;  // Copy original array to arr2 for sorting
    start = omp_get_wtime();  // Start the timer
    parallelBubbleSort(arr2);  // Perform parallel bubble sort
    end = omp_get_wtime();  // End the timer
    cout << "Parallel Bubble Sort: ";
    for (int x : arr2) {
        cout << x << " ";  // Output the sorted array
    }
    cout << "\nTime: " << fixed << setprecision(6) << (end - start) << " sec\n";  // Display the time taken

    // Sequential Merge Sort
    arr2 = arr;  // Copy original array to arr2 for sorting
    start = omp_get_wtime();  // Start the timer
    sequentialMergeSort(arr2, 0, n - 1);  // Perform sequential merge sort
    end = omp_get_wtime();  // End the timer
    cout << "Sequential Merge Sort: ";
    for (int x : arr2) {
        cout << x << " ";  // Output the sorted array
    }
    cout << "\nTime: " << fixed << setprecision(6) << (end - start) << " sec\n";  // Display the time taken

    // Parallel Merge Sort
    arr2 = arr;  // Copy original array to arr2 for sorting
    start = omp_get_wtime();  // Start the timer
    parallelMergeSort(arr2, 0, n - 1);  // Perform parallel merge sort
    end = omp_get_wtime();  // End the timer
    cout << "Parallel Merge Sort: ";
    for (int x : arr2) {
        cout << x << " ";  // Output the sorted array
    }
    cout << "\nTime: " << fixed << setprecision(6) << (end - start) << " sec\n";  // Display the time taken
}
