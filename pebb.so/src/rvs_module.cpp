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
#include "rvs_module.h"

#include <pci/pci.h>
#include <unistd.h>
#include <iostream>

#include "gpu_util.h"
#include "rvsloglp.h"
#include "worker.h"
#include "rvshsa.h"
#include "action.h"

/**
 * @defgroup PEBB PEBB Module
 *
 * @brief PCIe Bandwidth Benchmark Module
 *
 * The PCIe Bandwidth Benchmark attempts to saturate the PCIe bus with DMA
 * transfers between  * system memory and a target GPU card’s memory. The
 * maximum bandwidth obtained is reported  * to help debug low bandwidth issues.
 * The benchmark should be capable of targeting one, some or all of the GPUs
 * installed in a platform, reporting individual benchmark statistics for each.
 */

pebbworker* pworker;

int log(const char* pMsg, const int level) {
  return rvs::lp::Log(pMsg, level);
}


extern "C" void
rvs_module_get_version(int* Major, int* Minor, int* Revision) {
  *Major = BUILD_VERSION_MAJOR;
  *Minor = BUILD_VERSION_MINOR;
  *Revision = BUILD_VERSION_PATCH;
}

extern "C" int rvs_module_has_interface(int iid) {
  switch (iid) {
  case 0:
  case 1:
    return 1;
    }

  return 0;
}

extern "C" const char* rvs_module_get_name(void) {
  return "pebb";
}

extern "C" const char* rvs_module_get_description(void) {
  return "ROCm Validation Suite PEBB module";
}

extern "C" const char* rvs_module_get_config(void) {
  return "host_to_device (bool), device_to_host (bool), log_interval (integer)";
}

extern "C" const char* rvs_module_get_output(void) {
  return "interval_bandwidth (float array), bandwidth (float array)";
}

extern "C" int   rvs_module_init(void* pMi) {
//  pworker = nullptr;
  rvs::lp::Initialize(static_cast<T_MODULE_INIT*>(pMi));
  rvs::gpulist::Initialize();
  rvs::hsa::Init();
  return 0;
}

extern "C" int   rvs_module_terminate(void) {
  rvs::lp::Log("[module_terminate] pebb rvs_module_terminate() - entered",
               rvs::logtrace);
  if (pworker) {
    rvs::lp::Log(
      "[module_terminate] pebb rvs_module_terminate() - pworker exists",
                 rvs::logtrace);
    pworker->set_stop_name("module_terminate");
    pworker->stop();
    delete pworker;
    pworker = nullptr;
    rvs::lp::Log(
      "[module_terminate] pebb rvs_module_terminate() - monitoring stopped",
                 rvs::logtrace);
  }
  return 0;
}

extern "C" const char* rvs_module_get_errstring(int error) {
  switch (error) {
    default:
      return "General Error";
  }
}

extern "C" void* rvs_module_action_create(void) {
  return static_cast<void*>(new pebbaction);
}

extern "C" int   rvs_module_action_destroy(void* pAction) {
  delete static_cast<rvs::actionbase*>(pAction);
  return 0;
}

extern "C" int rvs_module_action_property_set(
  void* pAction, const char* Key, const char* Val) {
  return static_cast<rvs::actionbase*>(pAction)->property_set(Key, Val);
}

extern "C" int rvs_module_action_run(void* pAction) {
  return static_cast<rvs::actionbase*>(pAction)->run();
}


