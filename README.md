
# Particle Simulation: Concurrent vs. CUDA Parallelism

This project demonstrates the principles of **parallel programming** and **concurrent programming** in **C++**, using two different implementations of a 2D particle simulator, showcasing CPU-based Concurrency (using std::thread) and GPU-based Parallelism (using CUDA).

Both solutions accurately simulate particle movement, wall collisions, and maintain a count of total collisions within a defined world.

---

## Description

This project simulates the movement and collisions of a large number of round particles within a confined 2D Square space of 600*600 units. The simulation updates every particle's position based on it's own velocity which, as well as the starting position, is randomly generated using `random` module from C++.

Upon collision, the velocity is reversed (*-1) and the position of the particle is set again within boundaries. The total number of wall collisions is tracked at the end of the simulation in the CUDA implementation, but in the concurrent, it is updated in every thread process, since std::mutex allowed the changes on a total variable by multiple threads at the same time

---

## Algorithm

The simulation of different particles at the same time is a great candidate to test the performance of thread management both in the CPU and the GPU.

### Concurrent (CPU-Based):

This paradigm works well with the management of different tasks at the same time, but it's limited to the lesser amount of threads that are available in the CPU. That's why we use each thread to process a range of particles at a time.

With this conditions the program is able to run the simulation for a moderate number of particles using 16 threads in the tests made. This was efficient for the amount of particles and frames simulated, but this would definitely wouldn't be the option for a greater or even excesive amount of particles and frames.

### Parallel (GPU-Based):

For this paradigm, the things are completely different since it takes an enourmous advantage of the thousands of cores available in modern GPUs, which is what allows us to increase the number of particles massively. In this scenario each thread is used for not a group of particles but just one particle, making this approach the best for this simulator.

However, in this case, it's important to consider the cost or difficulties of handling data between the CPU and the GPU, the transfer of this memory can become a trouble if not handled correctly. That is the reaason why, in this code, the transfer of data is minimum, only being present in main parts of the code.

---

## Models

Both solutions follow a similar model, with the main difference being their execution environment and thread management.

### Core components

- Particle: The main entity, represented by its position, velocity, radius, and (in the CUDA solution) a collision counter.

- World: A defined 2D square boundary.

- Simulation Loop: An iterative process that updates the state of all particles for a time step over many frames.

- Collision Detection and Response: Logic to detect when a particle hits a wall and to update its velocity and position.

### Diagrams

#### Mathematical Backing:

For both paradigms we use basic Euler integration:

- $x_{t+dt} = x_t+v_x*dt$

- $y_{t+dt} = y_t+v_y*dt$

This is a common numerical integration method for simulating continuous motion in discrete time steps.

#### Theoretical Backing:

The use of std::mutex directly addresses the race condition problem in concurrent programming. This problem appears when multiple threads access and modify a shared resource (like the total collisions), leading to incorrect results. Mutexes provide mutual exclusion or resource locking, guaranteeing that only one thread can execute the critical section at any given time.

The *cudaMemcpy* operations represent the communication between host (CPU) and device (GPU). Minimizing these transfers (only copying data when necessary) is crucial for maximizing GPU acceleration. The *getCollisions* method explicitly copies data back, illustrating this.

#### Concurrent (CPU-Based)

**Main Thread:**
- Initialize `ParticleSystem` object.

**Simulation Loop (Per Frame):**
  - Call `updateFrame`.
    - **Inside `updateFrame`:**
      - Divide `particles` vector into `N` subsets.
      - **Create `N` Worker Threads (CPU):**
        - **Each Worker Thread:**
          - Executes `updateParticles` on its assigned particle subset.
          - **For each particle `p` in its subset:**
            - Update particle position.
            - **Check for Wall Collision (X):**
              - If `p.x` hits left/right wall:
                - Reverse velocity.
                - Update `p.x` to stay within bounds.
                - **Access the mutex for collisions.**
                - Increment `total_collisions`.
                - **Release the mutex.**
            - **Check for Wall Collision (Y):**
              - If `p.y` hits top/bottom wall:
                - Reverse velocity.
                - Update `p.y` to stay within bounds.
                - **Access the mutex for collisions.**
                - Increment `total_collisions`.
                - **Release the mutex.**
      - **Join All Worker Threads:**
        - Wait for all worker threads to complete their `updateParticles` execution.
  - Continue to the next frame.

**After All Frames:**
- Main Thread calculates total simulation time.
- Main Thread retrieves and reports `total_collisions`.

#### Parallel (GPU-Based)

**Host (CPU) Initialization:**
- Initialize `ParticleSystem` object.
- Allocate `device_particles` memory on the GPU using `cudaMalloc`.
- Copy initial `host_particles` data to `device_particles` using `cudaMemcpyHostToDevice`.

**Simulation Loop (Per Frame):**
  - **Host (CPU) calls `updateFrame`:**
    - **Launch CUDA Kernel (`updateParticles`):**
      - Passes `device_particles` and other simulation parameters to the GPU.
      - **GPU (Device) executes `updateParticles` Kernel:**
        - **Each GPU Thread:**
          - Processes a single `Particle` from `device_particles` at its `idx`, calculated for each Thread.
          - Update particle position.
          - **Check for Wall Collision (X-axis):**
            - If `p.x` hits left/right wall:
              - Reverse velocity.
              - Update `p.x` to stay within bounds.
              - Increment particle's individual counter.
          - **Check for Wall Collision (Y-axis):**
            - If `p.y` hits left/right wall:
              - Reverse velocity.
              - Update `p.y` to stay within bounds.
              - Increment particle's individual counter.
      - **All GPU Threads Complete.**
    - **Host (CPU) waits for GPU (`cudaDeviceSynchronize()`):**
      - Ensures all kernel computations for the frame are finished.
  - Continue to the next frame.

**After All Frames:**
- **Host (CPU) calls `getCollisions()`:**
  - Copy all `device_particles` data back to `host_particles` using `cudaMemcpyDeviceToHost`.
  - Sum the `collisions` member of all individual particles in `host_particles` to get the `total`.
- Host calculates total simulation time.
- Host reports total time and `total` wall collisions.
- Host frees `device_particles` memory on GPU using `cudaFree` in the destructor.

---

## Implementation

### Common:

- Object-Oriented Design using struct and class

- Constructor initialization

- Constants for world dimensions and time step

- Random number Generation

- Memory management (pre-allocation of memory and cudaMalloc)

- Timing

- Readability and Formatting ("cout" console log format)

### Concurrent:

- Use of Lambda Functions (used to give a function of the current object to a thread)

- Mutex protection to shared variables. (using lock_guard instead of lock and unlock)

### Parallel:

- Constructor marked for host and device

- Kernel grid / block configuration

- Thread ID calculation (linearization of 2D matrix)

- Bound checking on threads

- CUDA math functions for max and min values

- CUDA synchronizing between device and host

---

## Time and Space Complexity

### Concurrent:

#### Time Complexity:

- Initialization: O(P) where P is the number of particles, primarily for generating random data and filling the vector.

- Per Frame Update (updateFrame):

    - Each particle's update takes O(1) time.
    
    - Since N threads process P/N particles each (approximately), the ideal time for the parallel section would be O(P/N).
    
    - However, in reality, thread creation/joining, mutex contention (though minimal for simple increments), and context switching will add to this.
    
    - Therefore, the per-frame time complexity is approximately O(P/N+X), where X is the extra time mentioned before. In the worst case (single thread), it's O(P).

- Total Time: For F frames, the total time complexity is O(F⋅(P/N+X)).

#### Space Complexity:

- O(P) for storing the particles vector. Each particle object has a constant size.

- Additional O(N) for thread objects, which is minimun compared to particle data.

### Parallel:

#### Time Complexity:

- Initialization: O(P) on the host for generating random data.

- Host-to-Device Transfer: O(P) for cudaMemcpyHostToDevice.

- Per Frame Update (Kernel Launch):

    - Each GPU thread updates one particle in O(1) time.

- Device-to-Host Transfer (for getCollisions): O(P) for cudaMemcpyDeviceToHost. This occurs only when getCollisions() is called.

- Summing Collisions (on Host): O(P) for iterating through host_particles.

- Total Time: For F frames, the total time complexity is $O(P_{init}+P_{transfStart}+F+T_{sync}+P_{transfEnd}+P_{finalSum})$ where $T_{sync}$ is the overhead of cudaDeviceSynchronize().

#### Space Complexity:

- O(P) on the host for host_particles.

- O(P) on the device for device_particles.

---

## Benchmark

### Hardware:

- CPU: Intel Core i7-13620H 13th Gen (10 cores, 16 threads)

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU

### Software:

- OS: Windows 11 Microsoft Windows [Versión 22631.5413]

- Compiler: g++ 15.1.0, NVIDIA Cuda toolkit 12.9

- NVIDIA Driver: GeForce Game Ready driver 576.52 (compatible with CUDA toolkit 12.9)

---

## Building and Running

### Prerequisites

- C++11 (or later) compiler such as `g++`

- CUDA compiler as the one in CUDA toolkit

- Compatible GPU as GeForce Series NVIDIA GPUs

It is recommended to use the software and driver versions mentioned in Benchmark

### Compile

**Concurrent:**
```bash
g++ particleSim.cpp -o particleConc
```

**Parallel:**
```bash
nvcc particleSim.cu -o particleParallel
```

---

## Analysis

### Test cases

The following particle counts were used for 10 seconds (600 frames):

- Small	    1,000
- Medium    10,000
- Large     100,000
- X-Large   1,000,000

### Results

The main simulation was runned under the following conditions:

- Number of particles: 10,000

- Seconds of simulation: 60 (1 minute)

- Frames per second: 60

Here are the results:

**Concurrent**
```
=== CONCURRENT SIMULATION ===

Starting simulation for 60 seconds (3600 frames)

Testing with 10000 particles

Program running with 10000 particles operating in 16 threads

Progress: 10.00% (360/3600 frames)
Progress: 20.00% (720/3600 frames)
Progress: 30.00% (1080/3600 frames)
Progress: 40.00% (1440/3600 frames)
Progress: 50.00% (1800/3600 frames)
Progress: 60.00% (2160/3600 frames)
Progress: 70.00% (2520/3600 frames)
Progress: 80.00% (2880/3600 frames)
Progress: 90.00% (3240/3600 frames)
Progress: 100.00% (3600/3600 frames)

=== CONCURRENT SIMULATION COMPLETE ===

Total time: 7646 ms
Total wall collisions: 194097
Particles processed: 10000
```

**Parallel**
```
=== CUDA SIMULATION ===

Starting simulation for 60 seconds (3600 frames)

Testing with 10000 particles

Program running with 10000 particles on GPU
CUDA configuration: 40 blocks, 256 threads per block
Progress: 10.00% (360/3600 frames)
Progress: 20.00% (720/3600 frames)
Progress: 30.00% (1080/3600 frames)
Progress: 40.00% (1440/3600 frames)
Progress: 50.00% (1800/3600 frames)
Progress: 60.00% (2160/3600 frames)
Progress: 70.00% (2520/3600 frames)
Progress: 80.00% (2880/3600 frames)
Progress: 90.00% (3240/3600 frames)
Progress: 100.00% (3600/3600 frames)
=== PARALLEL SIMULATION COMPLETE ===

Total time: 66 ms
Total wall collisions: 193401
Particles processed: 10000
```

- Wall collisions: These values are similar, but not identical. This is expected due to the fact that particles were randomly generated for each simulation.

- Execution time: We went from 7646 ms to 66 ms, it decreased in a 99.13%, which is an exceptional improvement in performance

#### Conclusion:

For this particle simulation, the CUDA parallel solution is significantly faster for large-scale simulations, where the computational intensity on the GPU outweighs the overhead of memory transfers. The CPU-based concurrent solution is sufficient and potentially faster for much smaller datasets, where the setup costs of GPU computing are not justified.

---

## References:

- G. R. Lindfield and J. E. Penny, Numerical Methods: A Simple Introduction, 3rd ed. CRC Press, 2018.

- M. Herlihy and N. Shavit, The Art of Multiprocessor Programming, rev. 1st ed. Morgan Kaufmann, 2012.

- J. Nickolls, I. Buck, M. Garland, and K. Skadron, "Scalable Parallel Programming with CUDA," ACM Queue, vol. 6, no. 2, pp. 40–53, Apr. 2008.