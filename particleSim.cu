#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include <iomanip>

struct Particle
{
    float x, y;
    float vx, vy;
    float rad;
    int collisions;

    __host__ __device__ Particle(float x = 0, float y = 0,
                                 float vx = 0, float vy = 0,
                                 float rad = 2.0f,
                                 int col = 0) : x(x), y(y),
                                                vx(vx), vy(vy),
                                                rad(rad),
                                                collisions(col) {};
};

__global__ void updateParticles(Particle *particles, int particle_count,
                                float world_width, float world_height, float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < particle_count)
    {
        Particle &p = particles[idx];

        p.x += p.vx * dt;
        p.y += p.vy * dt;

        if (p.x - p.rad <= 0 || p.x + p.rad >= world_width)
        {
            p.vx = -p.vx;
            p.x = fmaxf(p.rad, fminf(world_width - p.rad, p.x));
            p.collisions++;
        }

        if (p.y - p.rad <= 0 || p.y + p.rad >= world_height)
        {
            p.vy = -p.vy;
            p.y = fmaxf(p.rad, fminf(world_height - p.rad, p.y));
            p.collisions++;
        }
    }
}

class ParticleSystem
{
private:
    std::vector<Particle> host_particles;
    Particle *device_particles;
    int particle_count;
    const float world_width = 600.0f;
    const float world_height = 600.0f;
    const float dt = 0.016f;

    int block_size = 256;
    int grid_size;

public:
    ParticleSystem(int count) : particle_count(count)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> pos_dist(50.0f, 550.0f);
        std::uniform_real_distribution<float> vel_dist(-200.0f, 200.0f);

        host_particles.reserve(particle_count);

        for (int i = 0; i < particle_count; i++)
        {
            host_particles.emplace_back(
                pos_dist(gen), pos_dist(gen),
                vel_dist(gen), vel_dist(gen));
        }

        grid_size = (particle_count + (block_size - 1)) / block_size;

        cudaMalloc(&device_particles, particle_count * sizeof(Particle));

        cudaMemcpy(device_particles, host_particles.data(),
                   particle_count * sizeof(Particle), cudaMemcpyHostToDevice);

        std::cout << "Program running with " << particle_count
                  << " particles on GPU\n";
        std::cout << "CUDA configuration: " << grid_size << " blocks, "
                  << block_size << " threads per block\n";
    }

    ~ParticleSystem()
    {
        cudaFree(device_particles);
    }

    void updateFrame()
    {
        updateParticles<<<grid_size, block_size>>>(
            device_particles, particle_count,
            world_width, world_height, dt);

        cudaDeviceSynchronize();
    }

    long long getCollisions()
    {
        // Copy particles back from GPU to CPU
        cudaMemcpy(host_particles.data(), device_particles,
                   particle_count * sizeof(Particle), cudaMemcpyDeviceToHost);

        long long total = 0;
        for (const auto &particle : host_particles)
        {
            total += particle.collisions;
        }
        return total;
    }

    int getParticleCount() { return particle_count; }
    float getWorldWidth() { return world_width; }
    float getWorldHeight() { return world_height; }
    float getTimeDiff() { return dt; }
};

int main()
{
    std::cout << "\n\n=== CUDA SIMULATION ===\n\n";

    int particle_count = 10000; // Start with more particles for GPU
    int seconds = 10;
    int frames = seconds * 60;
    std::cout << "Starting simulation for " << seconds
              << " seconds (" << frames << " frames)\n\n";
    std::cout << "Testing with " << particle_count << " particles\n\n";

    ParticleSystem system(particle_count);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int fr = 1; fr <= frames; fr++)
    {
        system.updateFrame();

        if (fr % 50 == 0)
        {
            float progress = (float)fr / frames * 100.0f;

            std::cout << "Progress: " << std::fixed << std::setprecision(2)
                      << progress << "% ("
                      << fr << "/" << frames << " frames)\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "=== PARALLEL SIMULATION COMPLETE ===\n\n";
    std::cout << "Total time: " << total_time << " ms\n";
    std::cout << "Total wall collisions: " << system.getCollisions() << "\n";
    std::cout << "Particles processed: " << particle_count << "\n\n\n";

    return 0;
}