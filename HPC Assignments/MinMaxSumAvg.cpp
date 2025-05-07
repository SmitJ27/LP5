/*

Step 1 - (if its already installed, go to step 2)
sudo apt-get update
sudo apt-get install g++

Step 2 - 
g++ -fopenmp -o parallel_reduction parallel_reduction.cpp

Step 3 - 
./parallel_reduction

sample input - 

Enter number of elements: 5
Enter elements: 
2.5
4.7
1.9
8.3
3.6

sample output - 

Minimum: 1.9
Maximum: 8.3
Sum: 21.0
Average: 4.200000

*/

#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main() {
    int n;
    // Prompt user for the number of elements
    cout << "Enter number of elements: ";
    cin >> n;

    vector<double> arr(n);
    cout << "Enter elements: " << endl;

    // Input the elements into the array
    for (double &x : arr) {
        cin >> x;
    }

    // Initialize variables for min, max, sum, and average
    double min_val = arr[0], max_val = arr[0], sum = 0.0, avg = 0.0;

    // Parallel loop to calculate min, max, and sum using OpenMP reductions
    // min_val, max_val, and sum are computed in parallel by using the reduction clause
    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val) reduction(+:sum)
    for (int i = 0; i < n; i++) {
        if (arr[i] < min_val) min_val = arr[i]; // Update min if necessary
        if (arr[i] > max_val) max_val = arr[i]; // Update max if necessary
        sum += arr[i]; // Accumulate sum of the elements
    }

    // Calculate the average after the parallel computation of sum
    avg = sum / n;

    // Output the computed values
    cout << "Minimum: " << min_val << endl; // Display minimum value
    cout << "Maximum: " << max_val << endl; // Display maximum value
    cout << "Sum: " << sum << endl; // Display sum of elements
    cout << "Average: " << avg << endl; // Display average value

    return 0;
}

