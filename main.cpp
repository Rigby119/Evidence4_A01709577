#include <iostream>
#include <vector>
#include <thread>
#include <numeric> // For accumulation
#include <mutex>   // For locking variables
#include <chrono>  // For timing results
#include <random>

using namespace std;

// SUM TOTAL
long long total_sum = 0;
// MUTEX FOR TOTAL
mutex sum_mutex;

// Function for threads
void parallel_sum_worker(const vector<int>& arr, size_t start, size_t end) {
    long long partial_sum = 0;
    for (size_t i = start; i < end; ++i) {
        partial_sum += arr[i];
    }

    sum_mutex.lock();
    total_sum += partial_sum;
    sum_mutex.unlock();
}

int main() {
    const size_t array_size = 100000000;
    vector<int> arr(array_size);
    cout << "Array size: " << array_size << endl;

    // This section should generate random numbers from 1-18
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1, 18);

    for (size_t i = 0; i < array_size; ++i) {
        arr[i] = distrib(gen);
    }

    // Time start
    auto start_normal = chrono::high_resolution_clock::now();

    long long sum = 0;

    for (int x : arr) {
        sum += x;
    }

    // Stop timer
    auto end_normal = chrono::high_resolution_clock::now();

    // Show time
    chrono::duration<double> duration_normal = end_normal - start_normal;
    cout << "Normal Sum: " << sum << endl;
    cout << "Normal Time: " << duration_normal.count() << " seconds" << endl;

    // Using threads
    auto start_thr = chrono::high_resolution_clock::now();

    // Determine threads
    int num_thr = 8;

    cout << "Using " << num_thr << " threads for parallel sum." << endl;
    vector<thread> threads;

    // Determine operational range for parallel
    size_t chunk_size = array_size / num_thr;
    size_t current_start = 0;

    for (int i = 0; i < num_thr; ++i) {
        size_t start = current_start;
        size_t end = current_start + chunk_size;

        // Last thread sums any remaining elements
        if (i == num_thr - 1) {
            end = array_size;
        }

        threads.emplace_back(parallel_sum_worker, cref(arr), start, end);
        current_start = end;
    }

    // Join threads
    for (thread& t : threads) {
        t.join();
    }

    // End timer
    auto end_thr = chrono::high_resolution_clock::now();

    // Show result time
    chrono::duration<double> duration_par = end_thr - start_thr;
    cout << "Parallel Sum:   " << total_sum << endl;
    cout << "Parallel Time:  " << duration_par.count() << " seconds" << endl;

    // Check if both sums match
    if (sum == total_sum) {
        cout << "SUCCESS! Sums match." << endl;
    } else {
        cout << "FAILED! Sums do not match." << endl;
        cout << "Difference: " << sum - total_sum << endl;
    }
    chrono::duration<double> duration_diff = duration_normal - duration_par;
    cout << "Time difference: " << duration_diff.count() << endl;

    return 0;
}