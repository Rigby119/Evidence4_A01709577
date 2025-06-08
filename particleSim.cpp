#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <mutex>

#include <iomanip>

struct Particle
{
    float x, y;
    float vx, vy;
    float rad;

    Particle(float x = 0, float y = 0,
             float vx = 0, float vy = 0,
             float rad = 2.0f,
             int col = 0) : x(x), y(y),
                            vx(vx), vy(vy),
                            rad(rad) {};
};

class ParticleSystem
{
private:
    std::vector<Particle> particles;
    const float world_width = 600.0f;
    const float world_height = 600.0f;
    const float dt = 0.016f;
    int num_threads;

    long long total_collisions = 0;
    std::mutex collision_mutex;

public:
    ParticleSystem(int particle_count, int threads = std::thread::hardware_concurrency())
        : num_threads(threads)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> pos_dist(50.0f, 550.0f);
        std::uniform_real_distribution<float> vel_dist(-200.0f, 200.0f);

        particles.reserve(particle_count);

        for (int i = 0; i < particle_count; ++i)
        {
            particles.emplace_back(
                pos_dist(gen), pos_dist(gen),
                vel_dist(gen), vel_dist(gen));
        }

        std::cout << "Program running with " << particle_count
                  << " particles operating in " << num_threads << " threads\n\n";
    }

    void updateParticles(int start_idx, int end_idx)
    {
        for (int i = start_idx; i < end_idx; i++)
        {
            Particle &p = particles[i];

            p.x += p.vx * dt;
            p.y += p.vy * dt;

            if (p.x - p.rad <= 0 || p.x + p.rad >= world_width)
            {
                p.vx = -p.vx;
                p.x = std::max(p.rad, std::min(world_width - p.rad, p.x));
                std::lock_guard<std::mutex> lock(collision_mutex);
                total_collisions++;
            }

            if (p.y - p.rad <= 0 || p.y + p.rad >= world_height)
            {
                p.vy = -p.vy;
                p.y = std::max(p.rad, std::min(world_height - p.rad, p.y));
                std::lock_guard<std::mutex> lock(collision_mutex);
                total_collisions++;
            }
        }
    }

    void updateFrame()
    {
        std::vector<std::thread> thr;

        const int particle_num = particles.size() / num_threads;

        for (int t = 0; t < num_threads; t++)
        {
            int start_idx = t * particle_num;
            int end_idx;
            if (t == num_threads - 1)
            {
                end_idx = particles.size();
            }
            else
            {
                end_idx = (t + 1) * particle_num;
            }

            thr.emplace_back([this, start_idx, end_idx]()
                             { this->updateParticles(start_idx, end_idx); });
        }

        for (auto &th : thr)
        {
            th.join();
        }
    }

    int getParticleCount() { return particles.size(); }
    float getWorldWidth() { return world_width; }
    float getWorldHeight() { return world_height; }
    float getTimeDiff() { return dt; }
    int getThreadCount() { return num_threads; }
    long long getCollisions() { return total_collisions; }
};

int main()
{
    std::cout << "\n\n=== CONCURRENT SIMULATION ===\n\n";

    int particle_count = 10000;
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

    std::cout << "\n=== CONCURRENT SIMULATION COMPLETE ===\n\n";
    std::cout << "Total time: " << total_time << " ms\n";
    std::cout << "Total wall collisions: " << system.getCollisions() << "\n";
    std::cout << "Particles processed: " << particle_count << "\n\n\n";

    return 0;
}