#pragma once

#include <cassert>
#include <chrono>
#include <string>
#include <unordered_map>

class Profiler
{
  struct Timer
  {
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
    bool started = false;
    size_t n_calls = 0;
    size_t total = 0;

    void start()
    {
      if(started == true) return; // already started
      started = true;
      m_start = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
      if(started == false) return; // nothing to stop
      const auto elapsed = std::chrono::high_resolution_clock::now() - m_start;
      started = false;
      total += std::chrono::duration<size_t, std::nano>(elapsed).count();
      n_calls ++;
    }

    void reset()
    {
      started = false;
      n_calls = 0;
      total = 0;
    }
  };

  std::unordered_map<std::string, Timer> timings;

public:

  Profiler()
  {
    timings.reserve(8);
  }

  void start(const std::string name)
  {
    timings[name].start();
  }

  void stop(const std::string name)
  {
    assert(timings.find(name) != timings.end());
    timings[name].stop();
  }

  void printStatAndReset()
  {
    size_t tot_nanoseconds = 0;
    for (auto &tm : timings)
    {
      tm.second.stop();
      tot_nanoseconds += tm.second.total;
    }

    const double nano2micro_factor = 1.0e-6;
    printf("Total time: %f ms\n", tot_nanoseconds * nano2micro_factor);
    printf("[Kernel    ]: tot Time (ms) | N executions | T (ms) per execution \n");

    for (auto &tm : timings)
    {
      const double microSecs = tm.second.total * nano2micro_factor;
      const double mSecPerCall = microSecs / tm.second.n_calls;
      printf("[%-10s]: %.07e | %12lu | %.07e\n", tm.first.c_str(), microSecs,
                                             tm.second.n_calls, mSecPerCall);
      tm.second.reset();
    }
  }
};

