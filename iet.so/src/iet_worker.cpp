/********************************************************************************
 *
 * Copyright (c) 2018 ROCm Developer Tools
 *
 * MIT LICENSE:
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "include/iet_worker.h"

#include <unistd.h>
#include <string>
#include <iostream>
#include <chrono>
#include <memory>

#include "rocm_smi/rocm_smi.h"

#include "include/blas_worker.h"
#include "include/log_worker.h"
#include "include/rvs_module.h"
#include "include/rvsloglp.h"

#define MODULE_NAME                             "iet"
#define POWER_PROCESS_DELAY                     5
#define MAX_MS_TRAIN_GPU                        1000
#define MAX_MS_WAIT_BLAS_THREAD                 10000
#define SGEMM_DELAY_FREQ_DEV                    10

#define IET_RESULT_PASS_MESSAGE                 "TRUE"
#define IET_RESULT_FAIL_MESSAGE                 "FALSE"

#define IET_BLAS_FAILURE                        "BLAS setup failed!"
#define IET_POWER_PROC_ERROR                    "could not get/process the GPU"\
                                                " power!"
#define IET_SGEMM_FAILURE                       "GPU failed to run the SGEMMs!"

#define IET_PWR_VIOLATION_MSG                   "power violation"
#define IET_PWR_TARGET_ACHIEVED_MSG             "target achieved"
#define IET_PWR_RAMP_EXCEEDED_MSG               "ramp time exceeded"
#define IET_PASS_KEY                            "pass"

#define IET_JSON_LOG_GPU_ID_KEY                 "gpu_id"
#define IET_MEM_ALLOC_ERROR                     1
#define IET_BLAS_ERROR                          2
#define IET_BLAS_MEMCPY_ERROR                   3
#define IET_BLAS_ITERATIONS                     25

using std::string;

bool IETWorker::bjson = false;

/**
 * @brief computes the difference (in milliseconds) between 2 points in time
 * @param t_end second point in time
 * @param t_start first point in time
 * @return time difference in milliseconds
 */
static uint64_t time_diff(
                std::chrono::time_point<std::chrono::system_clock> t_end,
                std::chrono::time_point<std::chrono::system_clock> t_start) {
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                            t_end - t_start);
    return milliseconds.count();
}

/**
 * @brief class default constructor
 */
IETWorker::IETWorker() {
    gpu_worker = nullptr;
    pwr_log_worker = nullptr;
}

IETWorker::~IETWorker() {
}


/**
 * @brief logs a message to JSON
 * @param key info type
 * @param value message to log
 * @param log_level the level of log (e.g.: info, results, error)
 */
void IETWorker::log_to_json(const std::string &key, const std::string &value,
                     int log_level) {
    if (IETWorker::bjson) {
        unsigned int sec;
        unsigned int usec;

        rvs::lp::get_ticks(&sec, &usec);
        void *json_node = rvs::lp::LogRecordCreate(MODULE_NAME,
                            action_name.c_str(), log_level, sec, usec);
        if (json_node) {
            rvs::lp::AddString(json_node, IET_JSON_LOG_GPU_ID_KEY,
                            std::to_string(gpu_id));
            rvs::lp::AddString(json_node, key, value);
            rvs::lp::LogRecordFlush(json_node);
        }
    }
}

/**
 * @brief performs the EDPp rampup on the given GPU (attempts to reach the given
 * target power)
 * @param err_description stores the error description if any
 * @return true if gpu training succeeded, false otherwise
 */
bool IETWorker::do_gpu_init_training(int gpuIdx,  uint64_t matrix_size, std::string  iet_ops_type){
    std::unique_ptr<rvs_blas> gpu_blas;
    rvs_blas  *free_gpublas;

    // setup rvsBlas
    gpu_blas = std::unique_ptr<rvs_blas>(new rvs_blas(gpuIdx,  matrix_size,  matrix_size,  matrix_size));

    //Hit the GPU with load to increase temperature
    for(int i = 0; i < IET_BLAS_ITERATIONS ; i++) {
         gpu_blas->run_blass_gemm(iet_ops_type);
    }

    free_gpublas = gpu_blas.release();
    delete free_gpublas;

    return true;
}

/**
 * @brief computes SGEMMs and power related statistics after the training stage
 */
void IETWorker::compute_gpu_stats(void) {
    float ms_per_sgemm, sgemm_target_power;
    float sgemm_target_power_si, total_ms_sgemm_si;

    // compute SGEMM time (ms)
    ms_per_sgemm = static_cast<float>(training_time_ms) / num_sgemms_training;
    // compute required number of SGEMM for the given target_power
    sgemm_target_power =
                    (target_power * num_sgemms_training) / avg_power_training;
    sgemm_target_power_si =
                    (sample_interval * sgemm_target_power) / training_time_ms;
    // compute the actual SGEMM frequency for the given target_power
    total_ms_sgemm_si = sgemm_target_power_si * ms_per_sgemm;
    sgemm_si_delay = sample_interval - total_ms_sgemm_si;
    if (sgemm_si_delay < 0) {
        sgemm_si_delay = 0;
    } else {
        if (sgemm_target_power_si > 0)
            sgemm_si_delay /= sgemm_target_power_si;
        else
            sgemm_target_power_si = sample_interval;

        sgemm_si_delay = sgemm_si_delay + sgemm_si_delay / SGEMM_DELAY_FREQ_DEV;
    }
}

/**
 * @brief computes the new SGEMM frequency so that the GPU will achieve the
 * given target_power
 * @param avg_power the last GPU average power over the last sample_interval
 */
void IETWorker::compute_new_sgemm_freq(float avg_power) {
    // compute the difference between the actual power data and the target_power
    float diff_power = avg_power - target_power;
    // gradually & dynamically increase/decrease the SGEMM frequency
    float sgemm_delay_dev = (abs(diff_power) * sgemm_si_delay) / target_power;
    if (diff_power < 0) {
        if (sgemm_si_delay - sgemm_delay_dev < 0)
            sgemm_si_delay = 1;
        else
            sgemm_si_delay -= sgemm_delay_dev;
    } else {
        sgemm_si_delay += sgemm_delay_dev;
    }
}


void blasThread(int gpuIdx,  uint64_t matrix_size, std::string  iet_ops_type, bool start )
{
    std::unique_ptr<rvs_blas> gpu_blas;
    rvs_blas  *free_gpublas;

    // setup rvsBlas
    gpu_blas = std::unique_ptr<rvs_blas>(new rvs_blas(gpuIdx,  matrix_size,  matrix_size,  matrix_size));

    //Hit the GPU with load to increase temperature
    for(int i = 0; i < IET_BLAS_ITERATIONS ; i++) {
         gpu_blas->run_blass_gemm(iet_ops_type);
    }

    start = false;

    free_gpublas = gpu_blas.release();
    delete free_gpublas;
}


/**
 * @brief performs the EDPp stress test on the given GPU (attempts to sustain
 * the target power)
 * @return true if EDPp test succeeded, false otherwise
 */
bool IETWorker::do_iet_power_stress(void) {
    std::chrono::time_point<std::chrono::system_clock> iet_start_time, end_time,
                                                        sampling_start_time;
    uint64_t  power_sampling_iters = 0;
    uint64_t  total_time_ms;
    uint64_t  last_avg_power;
    string    msg;
    float     cur_power_value;
    float     totalpower;
    float     avg_power;
    bool      result;
    bool      start;
   
    avg_power = 0;
    totalpower = 0;
    result = true;
    start = true;

    std::thread t(blasThread, gpu_device_index, matrix_size, iet_ops_type, start);
    t.detach();
 
    // record EDPp ramp-up start time
    iet_start_time = std::chrono::system_clock::now();

    for (;;) {
        // check if stop signal was received
        if (rvs::lp::Stopping())
            break;

       // get GPU's current average power
       rsmi_status_t rmsi_stat = rsmi_dev_power_ave_get(gpu_device_index, 0,
                                    &last_avg_power);

       if (rmsi_stat == RSMI_STATUS_SUCCESS) {
            cur_power_value = static_cast<float>(last_avg_power)/1e6;
       }

        msg = "[" + action_name + "] " + MODULE_NAME + " " +
                     std::to_string(gpu_id) + " " + " Target power is : " + " " + std::to_string(target_power);
        rvs::lp::Log(msg, rvs::logtrace);

        //check whether we reached the target power
        if((cur_power_value) >= target_power){
            msg = "[" + action_name + "] " + MODULE_NAME + " " +
                     std::to_string(gpu_id) + " " + " Average power met the target \
                     power quitting the test, current power is : " + " " + std::to_string(cur_power_value);
            rvs::lp::Log(msg, rvs::loginfo);
            result = true;
            break;
        }

        end_time = std::chrono::system_clock::now();

        total_time_ms = time_diff(end_time, iet_start_time);

        msg = "[" + action_name + "] " + MODULE_NAME + " " +
                     std::to_string(gpu_id) + " " + " Average power" + " " + std::to_string(cur_power_value);
        rvs::lp::Log(msg, rvs::loginfo);

        msg = "[" + action_name + "] " + MODULE_NAME + " " +
                     std::to_string(gpu_id) + " " + " Total time in ms " + " " + std::to_string(total_time_ms) +
                     " Run duration in ms " + " " + std::to_string(run_duration_ms);
        rvs::lp::Log(msg, rvs::logtrace);

        if (total_time_ms > run_duration_ms) {
            msg = "[" + action_name + "] " + MODULE_NAME + " " +
                     std::to_string(gpu_id) + " " + " Average power couldnt meet the target power  \
                     in the given interval, increase the duration and try again, \
                     Average power is :" + " " + std::to_string(cur_power_value);
            rvs::lp::Log(msg, rvs::loginfo);
            result = false;
            break;
	     }

       sleep(1000);

       // check if stop signal was received
       if (rvs::lp::Stopping())
         return true;
       }

       msg = "[" + action_name + "] " + MODULE_NAME + " " +
                   std::to_string(gpu_id) + " " + " End of worker thread " ;
       rvs::lp::Log(msg, rvs::loginfo);

       return result;
}


/**
 * @brief performs the Input EDPp test on the given GPU
 */
void IETWorker::run() {
    string msg, err_description;
    int error;

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
            std::to_string(gpu_id) + " start " + std::to_string(target_power);

    rvs::lp::Log(msg, rvs::loginfo);
    log_to_json("start", std::to_string(target_power), rvs::loginfo);

    if (run_duration_ms < MAX_MS_TRAIN_GPU)
        run_duration_ms += MAX_MS_TRAIN_GPU;

    bool pass = do_iet_power_stress();

    // check if stop signal was received
    if (rvs::lp::Stopping())
         return;

    msg = "[" + action_name + "] " + MODULE_NAME + " " +
               std::to_string(gpu_id) + " " + IET_PASS_KEY + ": " +
               (pass ? IET_RESULT_PASS_MESSAGE : IET_RESULT_FAIL_MESSAGE);
    rvs::lp::Log(msg, rvs::logresults);
}
