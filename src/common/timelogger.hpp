#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

typedef std::chrono::high_resolution_clock::time_point TimeVar;
/* #define duration(a)                                                            \ */
/*   ((double)std::chrono::duration_cast<std::chrono::microseconds>(a).count() /  \ */
/*    1e6) */
#define duration(a)                                                            \
  (std::chrono::duration_cast<std::chrono::microseconds>(a).count())
#define TimeNow() std::chrono::high_resolution_clock::now()

namespace YuriPerf {

/* Class for recording benchmarking times */
class TimeLogger {
public:
  TimeLogger() {}
  ~TimeLogger() {}

  // Record time for start of program
  void startProgram();

  // Record time for end of program
  void endProgram();

  // Start recording time for a given name
  void startRecording(std::string name);

  // Stop recording time for a given name
  void stopRecording(std::string name);

  // Stop recording time for the last name recorded
  void stopRecording();

  // Add a time to the list of times for a given name
  void addTime(std::string name, long duration, std::string time = "");

  // Get the average time for a given name
  double getAverageTime(std::string name);

  // Get the minimum time for a given name
  double getMinTime(std::string name);

  // Get the maximum time for a given name
  double getMaxTime(std::string name);

  // Get the standard deviation for a given name
  double getStdDev(std::string name);

  // Get the number of times a given name has been recorded
  size_t count(std::string name);

  // Clear all times
  void clear();
  // Clear times for a given name
  void clear(std::string name);

  // Print a histogram of the times
  void printHistogram();

  // Print detailed information, including average, min, max, and std dev
  void print();

  // Write a CSV file with the times
  void writeCSV(std::string filename);

  // Set whether or not to record times
  inline void setActive(bool active) { active_ = active; }

private:
  std::unordered_map<std::string, std::vector<std::pair<std::string, long>>> times_;

  std::unordered_map<std::string, TimeVar> start_times_;

  std::string prog_start_time_;
  std::string prog_end_time_;

  bool active_ = false;

  std::string last_name_;
};

} // namespace YuriPerf
