#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>

namespace hpcse {

class Timer {
public:
  Timer() {
    start_time.tv_sec  = 0;
    start_time.tv_usec = 0;
    stop_time.tv_sec   = 0;
    stop_time.tv_usec  = 0;
  }

  inline void start() {
    gettimeofday(&start_time, nullptr);
  }

  inline void stop() {
    gettimeofday(&stop_time, nullptr);
  }

  double get_timing() const {
    return (stop_time.tv_sec - start_time.tv_sec) + (stop_time.tv_usec - start_time.tv_usec)*1e-6;
  }

private:
  struct timeval start_time, stop_time;
};

}

#endif // TIMER_H
