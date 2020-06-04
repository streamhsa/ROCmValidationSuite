// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <mutex>

#define VERSION_STRING "3.4"

#include "include/rvs_memworker.h"
#include "include/Stream.h"
#include "include/HIPStream.h"
#include "include/rvs_memtest.h"
#include "include/rvsloglp.h"

extern rvs_memdata   memdata;

// Default size of 2^25
unsigned int ARRAY_SIZE = 33554432;
unsigned int num_times = 100;
unsigned int deviceIndex = 0;
bool use_float = false;
bool triad_only = false;
bool output_as_csv = false;
bool mibibytes = false;
std::string csv_separator = ",";

template <typename T>
void check_solution(const unsigned int ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, T& sum);

template <typename T>
void run_stress();

template <typename T>
void run_triad();

void parseArguments(int argc, char *argv[]);

void callBabel() {
  run_stress<float>();
  run_stress<double>();
  run_triad<float>();
  run_triad<double>();
}

template <typename T>
void run_stress()
{
  std::string   msg;

  if (!output_as_csv)
  {
    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Running kernels " + " block id :" + std::to_string(num_times);
    rvs::lp::Log(msg, rvs::loginfo);


    if (sizeof(T) == sizeof(float)) {
      msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Precision: float " + " \n";
      rvs::lp::Log(msg, rvs::loginfo);
    }
    else {
      msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Precision: double " + " \n";
      rvs::lp::Log(msg, rvs::loginfo);
    }


    if (mibibytes)
    {
      // MiB = 2^20
      msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Array size: " + std::to_string(ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0)) + " MiB" 
              + " (=" + std::to_string(ARRAY_SIZE*sizeof(T)*pow(2.0, -30.0)) + " GiB)" + "\n";
      rvs::lp::Log(msg, rvs::loginfo);

      msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Total size: " 
             + std::to_string(3.0*ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0)) + " MiB" +
                  " (=" + std::to_string(3.0*ARRAY_SIZE*sizeof(T)*pow(2.0, -30.0)) + " GiB)" + "\n";
      rvs::lp::Log(msg, rvs::loginfo);

    }
    else
    {
      // MB = 10^6
      msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Array size: " + 
              std::to_string(ARRAY_SIZE*sizeof(T)*1.0E-6) + " MB" +
              + " (=" + std::to_string(ARRAY_SIZE*sizeof(T)*1.0E-9) + " GB)" + "\n";
      rvs::lp::Log(msg, rvs::loginfo);

      msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + "Total size: " + std::to_string(3.0*ARRAY_SIZE*sizeof(T)*1.0E-6) + " MB"
                + " (=" + std::to_string(3.0*ARRAY_SIZE*sizeof(T)*1.0E-9) + " GB)" + "\n";
      rvs::lp::Log(msg, rvs::loginfo);
    }
  }

  // Create host vectors
  std::vector<T> a(ARRAY_SIZE);
  std::vector<T> b(ARRAY_SIZE);
  std::vector<T> c(ARRAY_SIZE);

  // Result of the Dot kernel
  T sum;

  Stream<T> *stream;

  // Use the HIP implementation
  stream = new HIPStream<T>(ARRAY_SIZE, deviceIndex);

  stream->init_arrays(startA, startB, startC);

  // List of times
  std::vector<std::vector<double>> timings(5);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // Main loop
  for (unsigned int k = 0; k < num_times; k++)
  {
    // Execute Copy
    t1 = std::chrono::high_resolution_clock::now();
    stream->copy();
    t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Mul
    t1 = std::chrono::high_resolution_clock::now();
    stream->mul();
    t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Add
    t1 = std::chrono::high_resolution_clock::now();
    stream->add();
    t2 = std::chrono::high_resolution_clock::now();
    timings[2].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Triad
    t1 = std::chrono::high_resolution_clock::now();
    stream->triad();
    t2 = std::chrono::high_resolution_clock::now();
    timings[3].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Dot
    t1 = std::chrono::high_resolution_clock::now();
    sum = stream->dot();
    t2 = std::chrono::high_resolution_clock::now();
    timings[4].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

  }

  // Check solutions
  stream->read_arrays(a, b, c);
  check_solution<T>(num_times, a, b, c, sum);

  // Display timing results
  if (output_as_csv)
  {
    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + 
      + "function" + csv_separator
      + "num_times" + csv_separator
      + "n_elements" + csv_separator
      + "sizeof" + csv_separator
      + ((mibibytes) ? "max_mibytes_per_sec" : "max_mbytes_per_sec") + csv_separator
      + "min_runtime" + csv_separator
      + "max_runtime" + csv_separator
      + "avg_runtime" + "\n";
    rvs::lp::Log(msg, rvs::loginfo);
  }else{

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " 
      +  "Function"
      +  ((mibibytes) ? "MiBytes/sec" : "MBytes/sec")
      +  "Min (sec)"
      +  "Max"
      +  "Average"
      + "\n"; 
    rvs::lp::Log(msg, rvs::loginfo);
  }

  std::string labels[5] = {"Copy", "Mul", "Add", "Triad", "Dot"};
  size_t sizes[5] = {
    2 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE
  };

  for (int i = 0; i < 5; i++)
  {
    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(timings[i].begin()+1, timings[i].end());

    // Calculate average; ignore the first result
    double average = std::accumulate(timings[i].begin()+1, timings[i].end(), 0.0) / (double)(num_times - 1);

    // Display results
    if (output_as_csv)
    {
       std::string t =  labels[i];
       msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + t
        + std::to_string(num_times) + csv_separator
        + std::to_string(ARRAY_SIZE) + csv_separator
        + std::to_string(sizeof(T)) + csv_separator
        + std::to_string(((mibibytes) ? pow(2.0, -20.0) : 1.0E-6) * sizes[i] / (*minmax.first)) + csv_separator
        + std::to_string(*minmax.first) + csv_separator
        + std::to_string(*minmax.second) + csv_separator
        + std::to_string(average)
        + "\n";
       rvs::lp::Log(msg, rvs::loginfo);
    }
    else
    {
      msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " 
        + labels[i]
        + std::to_string(((mibibytes) ? pow(2.0, -20.0) : 1.0E-6) * sizes[i] / (*minmax.first))
        + std::to_string(*minmax.first)
        + std::to_string(*minmax.second)
        + std::to_string(average)
        + "\n";
       rvs::lp::Log(msg, rvs::loginfo);
    }
  }

  delete stream;

}

template <typename T>
void run_triad()
{
  std::string msg;

  if (!output_as_csv)
  {
    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " +  "Running triad " + std::to_string(num_times) + " times" + "\n";
    rvs::lp::Log(msg, rvs::loginfo);

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " +  "Number of elements " + std::to_string(ARRAY_SIZE) + "\n";
    rvs::lp::Log(msg, rvs::loginfo);

    if (sizeof(T) == sizeof(float)) {
      msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " +  "Precision: float " + "\n";
      rvs::lp::Log(msg, rvs::loginfo);
    }
    else {
      msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " +  "Precision: double " + "\n";
      rvs::lp::Log(msg, rvs::loginfo);
    }

    if (mibibytes)
    {
       msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " + " Array size: " 
         + std::to_string(ARRAY_SIZE*sizeof(T)*pow(2.0, -10.0)) + " KiB" 
         + " (=" + std::to_string(ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0)) + " MiB)" + "\n";
       rvs::lp::Log(msg, rvs::loginfo);

      msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " +  "Total size: " 
        + std::to_string(3.0*ARRAY_SIZE*sizeof(T)*pow(2.0, -10.0)) + " KiB"
        + " (=" + std::to_string(3.0*ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0)) + " MiB)" + "\n";
      rvs::lp::Log(msg, rvs::loginfo);
    }
    else
    {
      msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " +  "Array size: " 
        + std::to_string(ARRAY_SIZE*sizeof(T)*1.0E-3) + " KB"
        + " (=" + std::to_string(ARRAY_SIZE*sizeof(T)*1.0E-6) + " MB)" + "\n";
      rvs::lp::Log(msg, rvs::loginfo);
    }
  }

  // Create host vectors
  std::vector<T> a(ARRAY_SIZE);
  std::vector<T> b(ARRAY_SIZE);
  std::vector<T> c(ARRAY_SIZE);

  Stream<T> *stream;

  // Use the HIP implementation
  stream = new HIPStream<T>(ARRAY_SIZE, deviceIndex);

  stream->init_arrays(startA, startB, startC);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // Run triad in loop
  t1 = std::chrono::high_resolution_clock::now();
  for (unsigned int k = 0; k < num_times; k++)
  {
    stream->triad();
  }
  t2 = std::chrono::high_resolution_clock::now();

  double runtime = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

  // Check solutions
  T sum = 0.0;
  stream->read_arrays(a, b, c);
  check_solution<T>(num_times, a, b, c, sum);

  // Display timing results
  double total_bytes = 3 * sizeof(T) * ARRAY_SIZE * num_times;
  double bandwidth = ((mibibytes) ? pow(2.0, -30.0) : 1.0E-9) * (total_bytes / runtime);

  if (output_as_csv)
  {
    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " 
      + "function" + csv_separator
      + "num_times" + csv_separator
      + "n_elements" + csv_separator
      + "sizeof" + csv_separator
      + ((mibibytes) ? "gibytes_per_sec" : "gbytes_per_sec") + csv_separator
      + "runtime"
      + "\n";
    rvs::lp::Log(msg, rvs::loginfo);

    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " 
      + "Triad" + csv_separator
      + std::to_string(num_times) + csv_separator
      + std::to_string(ARRAY_SIZE) + csv_separator
      + std::to_string(sizeof(T)) + csv_separator
      + std::to_string(bandwidth) + csv_separator
      + std::to_string(runtime)
      + "\n";
    rvs::lp::Log(msg, rvs::loginfo);
  }
  else
  {
    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " 
      + "--------------------------------"
      + "Runtime (seconds): " 
      + std::to_string(runtime) + "\n"
      + "Bandwidth (" + ((mibibytes) ? "GiB/s" : "GB/s") + "):  "
      + std::to_string(bandwidth) + "\n";
    rvs::lp::Log(msg, rvs::loginfo);
  }

  delete stream;
}

template <typename T>
void check_solution(const unsigned int ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, T& sum)
{
  // Generate correct solution
  T goldA = startA;
  T goldB = startB;
  T goldC = startC;
  T goldSum = 0.0;
  std::string  msg;

  const T scalar = startScalar;

  for (unsigned int i = 0; i < ntimes; i++)
  {
    // Do STREAM!
    if (!triad_only)
    {
      goldC = goldA;
      goldB = scalar * goldC;
      goldC = goldA + goldB;
    }
    goldA = goldB + scalar * goldC;
  }

  // Do the reduction
  goldSum = goldA * goldB * ARRAY_SIZE;

  // Calculate the average error
  double errA = std::accumulate(a.begin(), a.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldA); });
  errA /= a.size();
  double errB = std::accumulate(b.begin(), b.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldB); });
  errB /= b.size();
  double errC = std::accumulate(c.begin(), c.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldC); });
  errC /= c.size();
  double errSum = fabs(sum - goldSum);

  double epsi = std::numeric_limits<T>::epsilon() * 100.0;

  if (errA > epsi)
    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " 
      + "Validation failed on a[]. Average error " + std::to_string(errA)
      + "\n";
    rvs::lp::Log(msg, rvs::logerror);

  if (errB > epsi)
    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " 
      + "Validation failed on b[]. Average error " + std::to_string(errB)
      + "\n";
    rvs::lp::Log(msg, rvs::logerror);

  if (errC > epsi)
    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " 
      + "Validation failed on c[]. Average error " + std::to_string(errC)
      + "\n";
    rvs::lp::Log(msg, rvs::logerror);

  // Check sum to 8 decimal places
  if (!triad_only && errSum > 1.0E-8)
    msg = "[" + memdata.action_name + "] " + MODULE_NAME + " " 
      + "Validation failed on sum. Error " + std::to_string(errSum)
      + "\n"  
      + "Sum was " + std::to_string(sum) + " but should be " + std::to_string(goldSum)
      + "\n";
    rvs::lp::Log(msg, rvs::logerror);
}

int parseUInt(const char *str, unsigned int *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

void parseArguments(int argc, char *argv[])
{
  for (int i = 1; i < argc; i++)
  {
    if (!std::string("--list").compare(argv[i]))
    {
      listDevices();
      exit(EXIT_SUCCESS);
    }
    else if (!std::string("--device").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &deviceIndex))
      {
        std::cerr << "Invalid device index." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--arraysize").compare(argv[i]) ||
             !std::string("-s").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &ARRAY_SIZE))
      {
        std::cerr << "Invalid array size." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--numtimes").compare(argv[i]) ||
             !std::string("-n").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &num_times))
      {
        std::cerr << "Invalid number of times." << std::endl;
        exit(EXIT_FAILURE);
      }
      if (num_times < 2)
      {
        std::cerr << "Number of times must be 2 or more" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--float").compare(argv[i]))
    {
      use_float = true;
    }
    else if (!std::string("--triad-only").compare(argv[i]))
    {
      triad_only = true;
    }
    else if (!std::string("--csv").compare(argv[i]))
    {
      output_as_csv = true;
    }
    else if (!std::string("--mibibytes").compare(argv[i]))
    {
      mibibytes = true;
    }
    else if (!std::string("--help").compare(argv[i]) ||
             !std::string("-h").compare(argv[i]))
    {
      std::cout << std::endl;
      std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  -h  --help               Print the message" << std::endl;
      std::cout << "      --list               List available devices" << std::endl;
      std::cout << "      --device     INDEX   Select device at INDEX" << std::endl;
      std::cout << "  -s  --arraysize  SIZE    Use SIZE elements in the array" << std::endl;
      std::cout << "  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)" << std::endl;
      std::cout << "      --float              Use floats (rather than doubles)" << std::endl;
      std::cout << "      --triad-only         Only run triad" << std::endl;
      std::cout << "      --csv                Output as csv table" << std::endl;
      std::cout << "      --mibibytes          Use MiB=2^20 for bandwidth calculation (default MB=10^6)" << std::endl;
      std::cout << std::endl;
      exit(EXIT_SUCCESS);
    }
    else
    {
      std::cerr << "Unrecognized argument '" << argv[i] << "' (try '--help')"
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}
