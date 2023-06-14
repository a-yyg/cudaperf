#include "timelogger.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include <cmath>

namespace YuriPerf {

void TimeLogger::startRecording(std::string name) {
  if (!active_)
    return;

  start_times_[name] = TimeNow();
  last_name_ = name;
}

void TimeLogger::stopRecording(std::string name) {
  if (!active_)
    return;

  auto start_time = start_times_[name];
  auto end_time = TimeNow();
  addTime(name, duration(end_time - start_time));
}

void TimeLogger::stopRecording() {
  if (!active_ || last_name_.empty())
    return;

  auto start_time = start_times_[last_name_];
  auto end_time = TimeNow();
  addTime(last_name_, duration(end_time - start_time));
}

void TimeLogger::addTime(std::string name, long time) {
  if (!active_)
    return;

  times_[name].push_back(time);
}

double TimeLogger::getAverageTime(std::string name) {
  if (!active_)
    return 0;

  double sum = 0;
  for (auto &time : times_[name]) {
    sum += time;
  }
  return sum / times_[name].size();
}

double TimeLogger::getMinTime(std::string name) {
  if (!active_)
    return 0;

  double min = times_[name][0];
  for (auto &time : times_[name]) {
    if (time < min)
      min = time;
  }
  return min;
}

double TimeLogger::getMaxTime(std::string name) {
  if (!active_)
    return 0;

  double max = times_[name][0];
  for (auto &time : times_[name]) {
    if (time > max)
      max = time;
  }
  return max;
}

double TimeLogger::getStdDev(std::string name) {
  if (!active_)
    return 0;

  double avg = getAverageTime(name);
  double sum = 0;
  for (auto &time : times_[name]) {
    sum += (time - avg) * (time - avg);
  }
  return std::sqrt(sum / times_[name].size());
}

size_t TimeLogger::count(std::string name) {
  if (!active_)
    return 0;

  return times_[name].size();
}

void TimeLogger::clear() {
  if (!active_)
    return;

  times_.clear();
}

void TimeLogger::clear(std::string name) {
  if (!active_)
    return;

  times_[name].clear();
}

void TimeLogger::printHistogram() {
  if (!active_)
    return;

  // order by time
  std::vector<std::pair<std::string, double>> times;
  double total = 0;
  for (auto &time : times_) {
    double avg_time = getAverageTime(time.first);
    times.push_back(std::make_pair(time.first, avg_time));
    total += avg_time;
  }

  /* std::sort(times.begin(), times.end(), */
  /*           [](const std::pair<std::string, double> &a, */
  /*              const std::pair<std::string, double> &b) { */
  /*             return a.second > b.second; */
  /*           }); */

  std::cout << "--- Histogram ---" << std::endl;
  for (auto &time : times) {
    std::cout << std::setw(25) << std::left << time.first << ": "
              << std::setw(15) << std::left << time.second << " us"
              << " (" << time.second / total * 100 << "%)" << std::endl;
  }
  std::cout << "Total: " << total * 1000 << " ms" << std::endl;
}

void TimeLogger::print() {
  if (!active_)
    return;

  std::cout << "--- Count ---" << std::endl;
  for (auto &time : times_) {
    std::cout << std::setw(25) << std::left << time.first << ": "
              << std::setw(15) << std::left << count(time.first) << std::endl;
  }

  // Print histogram, then avg, ranges and std dev
  printHistogram();

  std::cout << "--- Average ---" << std::endl;
  for (auto &time : times_) {
    std::cout << std::setw(25) << std::left << time.first << ": "
              << std::setw(15) << std::left << getAverageTime(time.first)
              << " us" << std::endl;
  }

  std::cout << "--- Ranges ---" << std::endl;
  for (auto &time : times_) {
    std::cout << std::setw(25) << std::left << time.first << ": "
              << std::setw(15) << std::left << getMinTime(time.first)
              << " us - " << std::setw(15) << std::left
              << getMaxTime(time.first) << " us" << std::endl;
  }

  std::cout << "--- Std Dev ---" << std::endl;
  for (auto &time : times_) {
    std::cout << std::setw(25) << std::left << time.first << ": "
              << std::setw(15) << std::left << getStdDev(time.first)
              << " us" << std::endl;
  }
}

} // namespace YuriPerf
