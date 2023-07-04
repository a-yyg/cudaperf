#include "timelogger.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>

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

  // get current time in format HH:MM:SS
  time_t now = time(0);
  tm *ltm = localtime(&now);
  std::string time = std::to_string(ltm->tm_hour) + ":" +
                     std::to_string(ltm->tm_min) + ":" +
                     std::to_string(ltm->tm_sec);

  addTime(name, duration(end_time - start_time), time);
}

void TimeLogger::stopRecording() {
  if (!active_ || last_name_.empty())
    return;

  auto start_time = start_times_[last_name_];
  auto end_time = TimeNow();
  time_t now = time(0);

  // get current time in format HH:MM:SS
  tm *ltm = localtime(&now);
  std::string time = std::to_string(ltm->tm_hour) + ":" +
                     std::to_string(ltm->tm_min) + ":" +
                     std::to_string(ltm->tm_sec);

  addTime(last_name_, duration(end_time - start_time), time);
}

void TimeLogger::addTime(std::string name, long duration, std::string time) {
  if (!active_)
    return;

  times_[name].push_back(std::make_pair(time, duration));
}

double TimeLogger::getAverageTime(std::string name) {
  if (!active_)
    return 0;

  double sum = 0;
  for (auto &time : times_[name]) {
    sum += time.second;
  }
  return sum / times_[name].size();
}

double TimeLogger::getMinTime(std::string name) {
  if (!active_)
    return 0;

  double min = times_[name][0].second;
  for (auto &time : times_[name]) {
    if (time.second < min)
      min = time.second;
  }
  return min;
}

double TimeLogger::getMaxTime(std::string name) {
  if (!active_)
    return 0;

  double max = times_[name][0].second;
  for (auto &time : times_[name]) {
    if (time.second > max)
      max = time.second;
  }
  return max;
}

double TimeLogger::getStdDev(std::string name) {
  if (!active_)
    return 0;

  double avg = getAverageTime(name);
  double sum = 0;
  for (auto &time : times_[name]) {
    sum += (time.second - avg) * (time.second - avg);
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
  /* printHistogram(); */

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
              << std::setw(15) << std::left << getStdDev(time.first) << " us"
              << std::endl;
  }
}

void TimeLogger::writeCSV(std::string filename) {
  if (!active_)
    return;

  std::ofstream file(filename);
  // Write header with each name
  file << "time,";
  file << "iteration,";
  for (auto &time : times_) {
    file << time.first << ",";
  }
  
  file << "total";

  file << std::endl;

  // Write each time
  for (size_t i = 0; i < times_.begin()->second.size(); i++) {
    long total = 0;

    file << times_.begin()->second[i].first << ",";
    file << i << ",";
    for (auto &time : times_) {
      file << time.second[i].second << ",";
      total += time.second[i].second;
    }

    file << total;

    file << std::endl;
  }
  file.close();
}

} // namespace YuriPerf
