
# Evidence4_A01709577

This project demonstrates the principles of **parallel programming** in **C++**, using `std::thread` and `std::mutex` to significantly speed up the summation of a large array of numbers. It compares the performance of a traditional sequential approach against a multi-threaded parallel implementation, highlighting the benefits and challenges of concurrent execution.

---

## Description

This project solves a performance bottleneck encountered when summing a large array of numbers — a common task in data analysis, scientific computing, and financial modeling. The goal is to improve execution time through **parallel programming**, leveraging multi-core CPU architecture.

The problem is useful because it reflects real-world challenges where processing large datasets efficiently is critical. The project explores how concurrency can be applied to divide and conquer the summation task.

---

## Models

The solution follows the **parallel programming paradigm**. The logic is as follows:

- The array is divided into `P` equal chunks (`P` = number of threads).
- Each chunk is processed by a separate thread.
- Each thread calculates a **partial sum** of its chunk.
- A mutex (`std::mutex`) ensures safe updating of the shared `total_sum`.

### Diagram

[ Main Thread ]  
|  
|---> [ Thread 1 ] --> Partial Sum 1  
|---> [ Thread 2 ] --> Partial Sum 2 |  
...
|---> [ Thread N-1 ] --> Partial Sum N-1 | 
|---> [ Thread N ] --> Combine --> total_sum /

---

## Implementation

This project is implemented in **C++11** using the `std::thread` and `std::mutex` standard library features.

Main concepts used:

- `std::thread`: For concurrent execution.
- `std::mutex`: To prevent race conditions.
- Array of size 100,000,000 with random values from 1 to 18.
- Parallel vs Sequential performance comparison.

---

## Test

The code has random generation so each time it runs, a random test is executed.
For 20 tests the output was the same, success, i've also tried greater sizes for the array of numbers, which also succeded.

Example output includes:
- Total sum
- Time taken for both methods
- A "SUCCESS!" check for validation

---

## Analysis

### Time Complexity

- **Sequential Sum**: `O(N)`
- **Parallel Sum**: Ideal case: `O(N/P)`, where `P` = number of threads

---

## Building and Running

### Prerequisites

- C++11 (or later) compiler such as `g++` or `clang++`

### Compile

```bash
g++ main.cpp -o main -std=c++11 -pthread
