
@section ugs1 1 Introduciton
The ROCm Validation Suite (RVS) is a system administrator’s and cluster
manager's tool for detecting and troubleshooting common problems affecting AMD
GPU(s) running in a high-performance computing environment, enabled using the
ROCm software stack on a compatible platform.

The RVS is a collection of tests, benchmarks and qualification tools each
targeting a specific sub-system of the ROCm platform. All of the tools are
implemented in software and share a common command line interface. Each set of
tests are implemented in a “module” which is a library encapsulating the f
unctionality specific to the tool. The CLI can specify the directory containing
modules to use when searching for libraries to load. Each module may have a set
of options that it defines and a configuration file that supports its execution.

@section usg2 2 Installing RVS

RVS cab be obtained by building it fro source code base or by installing from
pre-built package.

@subsection ugs21 2.1 Building from Source Code

RVS has been developed as open source solution. Its source code and belonging
documentation can be found at AMD's GitHub page.\n
In order to build RVS from source code please visit
[ROCm Validation Suite GitHub
site](https://github.com/ROCm-Developer-Tools/ROCmValidationSuite)
and follow instructions in README file.

@subsection usg22 2.2 Installing from Package
Please download `rocm-validation-suite-$(RVSVER).deb` or `.rpt` file from AMD
site. Install package using your favorite package manager.

RVS components is installed in `/opt/rocm/rvs`. Package contains:
- executable modules
- user guide (located in in _install-base_/userguide/html)
- man page (located in _install-base_/man)
- configuration examples (located in _install-base_/conf)

If needed, you may remove RVS package using you favorite package manager.


@section usg3 3 Basic Concepts

@subsection usg31 3.1 RVS Architecture

RVS is implemented as a set of modules each implementing particular test
functionality. Modules are invoked from one central place (aka Launcher) which
is responsible for reading input (command line and test configuration file),
loading and running appropriate modules and providing test output. RVS
architecture is built around concept of Linux shared objects, thus
allowing for easy addition of new modules in the future.


@subsection usg31a 3.2 Available Modules

@subsubsection usg31a01 3.2.1 GPU Properties – GPUP
The GPU Properties module queries the configuration of a target device and
returns the device’s static characteristics.\n
These static values can be used to debug issues such as device support,
performance and firmware problems.

@subsubsection usg31a02 3.2.2 GPU Monitor – GM module
The GPU monitor tool is capable of running on one, some or all of the GPU(s)
installed and will report various information at regular intervals. The module
can be configured to halt another RVS modules execution if one of the quantities
exceeds a specified boundary value.
@subsubsection usg31a03 3.2.3 PCI Express State Monitor  – PESM module
The PCIe State Monitor tool is used to actively monitor the PCIe interconnect
between the host platform and the GPU. The module will register a “listener” on
a target GPU’s PCIe interconnect, and log a message whenever it detects a state
change. The PESM is able to detect the following state changes:

1.  PCIe link speed changes
2.  GPU power state changes

@subsubsection usg31a04 3.2.4 ROCm Configuration Qualification Tool - RCQT
module
The ROCm Configuration Qualification Tool ensures the platform is capable of
running ROCm applications and is configured correctly. It checks the installed
versions of the ROCm components and the platform configuration of the system.
This includes checking that dependencies, corresponding to the associated
operating system and runtime environment, are installed correctly. Other
qualification steps include checking:

1.  The existence of the /dev/kfd device
2.  The /dev/kfd device’s permissions
3.  The existence of all required users and groups that support ROCm
4.  That the user mode components are compatible with the drivers, both the KFD
and the amdgpu driver.
5.  The configuration of the runtime linker/loader qualifying that all ROCm
libraries are in the correct search path.

@subsubsection usg31a05 3.2.5 PCI Express Qualification Tool – PEQT module
The PCIe Qualification Tool consists is used to qualify the PCIe bus on which
the GPU is connected. The qualification test is capable of determining the
following characteristics of the PCIe bus interconnect to a GPU:

1.  Support for Gen 3 atomic completers
2.  DMA transfer statistics
3.  PCIe link speed
4.  PCIe link width

@subsubsection usg31a06 3.2.6 SBIOS Mapping Qualification Tool – SMQT module
The GPU SBIOS mapping qualification tool is designed to verify that a
platform’s SBIOS has satisfied the BAR mapping requirements for VDI and Radeon
Instinct products for ROCm support.

Refer to the “ROCm Use of Advanced PCIe Features and Overview of How BAR Memory
is Used In ROCm Enabled System” web page for more information about how BAR
memory is initialized by VDI and Radeon products.

@subsubsection usg31a07 3.2.7 P2P Benchmark and Qualification Tool – PBQT module
The P2P Benchmark and Qualification Tool  is designed to provide the list of all
GPUs that support P2P and characterize the P2P links between peers. In addition
to testing for P2P compatibility, this test will perform a peer-to-peer
throughput test between all P2P pairs for performance evaluation. The P2P
Benchmark and Qualification Tool will allow users to pick a collection of two or
more GPUs on which to run. The user will also be able to select whether or not
they want to run the throughput test on each of the pairs.

Please see the web page “ROCm, a New Era in Open GPU Computing” to find out more
about the P2P solutions available in a ROCm environment.

@subsubsection usg31a08 3.2.8 PCI Express Bandwidth Benchmark – PEBB module
The PCIe Bandwidth Benchmark attempts to saturate the PCIe bus with DMA
transfers between system memory and a target GPU card’s memory. The maximum
bandwidth obtained is reported to help debug low bandwidth issues. The
benchmark should be capable of  targeting one, some or all of the GPUs
installed in a platform, reporting individual benchmark statistics for each.

@subsubsection usg31a09 3.2.9 GPU Stress Test  - GST module
The GPU Stress Test runs a Graphics Stress test or SGEMM/DGEMM
(Single/Double-precision General Matrix Multiplication) workload on one, some or
all GPUs. The GPUs can be of the same or different types. The duration of the
benchmark should be configurable, both in terms of time (how long to run) and
iterations (how many times to run).

The test should be capable driving the power level equivalent to the rated TDP
of the card, or levels below that. The tool must be capable of driving cards at
TDP-50% to TDP-100%, in 10% incremental jumps. This should be controllable by
the user.

@subsubsection usg31a10 3.2.10 Input EDPp Test  - IET module
The Input EDPp Test generates EDP peak power on all input rails. This test is
used to verify if the system PSU is capable of handling the worst case power
spikes of the board.  Peak Current at defined period  =  1 minute moving
average power.


@subsection usg32 3.2 Configuration Files

The RVS tool will allow the user to indicate a configuration file, adhering to
the YAML 1.2 specification, which details the validation tests to run and the
expected results of a test, benchmark or configuration check.

The configuration
file used for an execution is specified using the `--config` option. The default
configuration file used for a run is `rvs.conf`, which will include default
values for all defined tests, benchmarks and configurations checks, as well as
device specific configuration values. The format of the configuration files
determines the order in which actions are executed, and can provide the number
of times the test will be executed as well.

Configuration file is, in YAML terms, mapping of 'actions' keyword into
sequence of action items. Action items are themselves YAML keyed lists. Each
list consists of several _key:value_ pairs. Some keys may have values which
are keyed lists themselves (nested mappings).

Action item (or action for short) uses keys to define nature of validation test
to be performed. Each action has some common keys -- like 'name', 'module',
'deviceid' -- and test specific keys which depend on the module being used.

An example of RVS configuration file is given here:


    actions:
    - name: action_1
      device: all
      module: gpup
      properties:
        mem_banks_count:
      io_links-properties:
        version_major:
    - name: action_2
      module: gpup
      device: all
      properties:
        mem_banks_count:
    - name: action_3
    ...


@subsection usg33 3.3 Common Configuration Keys

Common configuration keys applicable to most module are summarized in the
table below:\n
<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>name</td><td>String</td><td>The name of the defined action.</td></tr>
<tr><td>device</td><td>Collection of String</td>
<td>This is a list of device indexes (gpu ids), or the keyword “all”. The
defined actions will be executed on the specified device, as long as the action
targets a device specifically (some are platform actions). If an invalid device
id value or no value is specified the tool will report that the device was not
found and terminate execution, returning an error regarding the configuration
file.</td></tr>

<tr><td>deviceid</td><td>Integer</td><td>This is an optional parameter, but if
specified it restricts the action to a specific device type
corresponding to the deviceid.</td></tr>
<tr><td>parallel</td><td>Bool</td><td>If this key is false, actions will be run
on one device at a time, in the order specified in the device list, or the
natural ordering if the device value is “all”. If this parameter is true,
actions will be run on all specified devices in parallel. If a value isn’t
specified the default value is false.</td></tr>

<tr><td>count</td><td>Integer</td><td>This specifies number of times to execute
the action. If the value is 0, execution will continue indefinitely. If a value
isn’t specified the default is 1. Some modules will ignore this
parameter.</td></tr>

<tr><td>wait</td><td>Integer</td><td>This indicates how long the test should
wait
between executions, in milliseconds. Some
modules will ignore this parameter. If the
count key is not specified, this key is ignored.
duration Integer This parameter overrides the count key, if
specified. This indicates how long the test
should run, given in milliseconds. Some
modules will ignore this parameter.</td></tr>


<tr><td>module</td><td>String</td><td>This parameter specifies the module that
will be used in the execution of the action. Each module has a set of sub-tests
or sub-actions that can be configured based on its specific
parameters.</td></tr>
</table>

@subsection usg34 3.4 Command Line Options

Command line options are summarized in the table below:

<table>
<tr><th>Short option</th><th>Long option</th><th> Description</th></tr>
<tr><td>-a</td><td>\-\-appendLog</td><td>When generating a debug logfile,
do not overwrite the contents
of a current log. Used in conjuction with the -d and -l options.
</td></tr>

<tr><td>-c</td><td>\-\-config</td><td>Specify the configuration file to be used.
The default is \<installbase\>/RVS/conf/RVS.conf
</td></tr>

<tr><td></td><td>\-\-configless</td><td>Run RVS in a configless mode.
Executes a "long" test on all supported GPUs.</td></tr>

<tr><td>-d</td><td>\-\-debugLevel</td><td>Specify the debug level for the output
log. The range is 0 to 5 with 5 being the most verbose.
Used in conjunction with the -l flag.</td></tr>

<tr><td>-g</td><td>\-\-listGpus</td><td>List the GPUs available and exit.
This will only list GPUs that are supported by RVS.</td></tr>

<tr><td>-i</td><td>\-\-indexes</td><td>Comma separated list of  devices to run
RVS on. This will override the device values specified in the configuration file
for every action in the configuration file, including the "all" value.</td></tr>

<tr><td>-j</td><td>\-\-json</td><td>Output should use the JSON format.</td></tr>

<tr><td>-l</td><td>\-\-debugLogFile</td><td>Specify the logfile for debug
information. This will produce a log file intended for post-run analysis after
an error.</td></tr>

<tr><td></td><td>\-\-quiet</td><td>No console output given. See logs and return
code for errors.</td></tr>

<tr><td>-m</td><td>\-\-modulepath</td><td>Specify a custom path for the RVS
modules.</td></tr>

<tr><td></td><td>\-\-specifiedtest</td><td>Run a specific test in a configless
mode. Multiple word tests should be in quotes. This action will default to all
devices, unless the \-\-indexes option is specifie.</td></tr>

<tr><td>-t</td><td>\-\-listTests</td><td>List the modules available to be
executed through RVS and exit. This will list only the readily loadable modules
given the current path and library conditions.</td></tr>

<tr><td>-v</td><td>\-\-verbose</td><td>Enable verbose reporting. This is
equivalent to specifying the -d 5 option.</td></tr>

<tr><td></td><td>\-\-version</td><td>Displays the version information and exits.
</td></tr>

<tr><td>-h</td><td>\-\-help</td><td>Display usage information and exit.
</td></tr>

</table>

@section usg4 4 GPUP Module
The GPU properties module provides an interface to easily dump the static
characteristics of a GPU. This information is stored in the sysfs file system
for the kfd, with the following path:

    /sys/class/kfd/kfd/topology/nodes/<node id>

Each of the GPU nodes in the directory is identified with a number,
indicating the device index of the GPU. This module will ignore count, duration
or wait key values.

@subsection usg41 4.1 Module Specific Keys
<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>properties</td><td>Collection of Strings</td>
<td>The properties key specifies what configuration property or properties the
query is interested in. Possible values are:\n
all - collect all settings\n
gpu_id\n
cpu_cores_count\n
simd_count\n
mem_banks_count\n
caches_count\n
io_links_count\n
cpu_core_id_base\n
simd_id_base\n
max_waves_per_simd\n
lds_size_in_kb\n
gds_size_in_kb\n
wave_front_size\n
array_count\n
simd_arrays_per_engine\n
cu_per_simd_array\n
simd_per_cu\n
max_slots_scratch_cu\n
vendor_id\n
device_id\n
location_id\n
drm_render_minor\n
max_engine_clk_fcompute\n
local_mem_size\n
fw_version\n
capability\n
max_engine_clk_ccompute\n
</td></tr>
<tr><td>io_links-properties</td><td>Collection of Strings</td>
<td>The properties key specifies what configuration
property or properties the query is interested in.
Possible values are:\n
all - collect all settings\n
count - the number of io_links\n
type\n
version_major\n
version_minor\n
node_from\n
node_to\n
weight\n
min_latency\n
max_latency\n
min_bandwidth\n
max_bandwidth\n
recommended_transfer_size\n
flags\n
</td></tr>
</table>
@subsection usg42 4.2 Output

Module specific output keys are described in the table below:
<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>properties-values</td><td>Collection of Integers</td>
<td>The collection will contain a positive integer value for each of the valid
properties specified in the properties config key.</td></tr>
<tr><td>io_links-propertiesvalues</td><td>Collection of Integers</td>
<td>The collection will contain a positive integer value for each of the valid
properties specified in the io_links-properties config key.</td></tr>
</table>
Each of the settings specified has a positive integer value. For each
setting requested in the properties key a message with the following format will
be returned:

    [RESULT][<timestamp>][<action name>] gpup <gpu id> <property> <property value>

For each setting in the io_links-properties key a message with the following
format will be returned:

    [RESULT][<timestamp>][<action name>] gpup <gpu id> <io_link id> <property> <property value>

@subsection usg43 4.3 Examples

@section usg5 5 GM Module
The GPU monitor module can be used monitor and characterize the response of a
GPU to different levels of use. This module is intended to run concurrently with
other actions, and provides a ‘start’ and ‘stop’ configuration key to start the
monitoring and then stop it after testing has completed. The module can also be
configured with bounding box values for interested GPU parameters. If any of the
GPU’s parameters exceed the bounding values on a specific GPU an INFO warning
message will be printed to stdout while the bounding value is still exceeded.

@subsection usg51 5.1 Module Specific Keys

<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>monitor</td><td>Bool</td>
<td>If this this key is set to true, the GM module will start monitoring on
specified devices. If this key is set to false, all other keys are ignored and
monitoring of the specified device will be stopped.</td></tr>
<tr><td>metrics</td>
<td>Collection of Structures, specifying the metric, if there are bounds and the
bound values. The structures have the following format:\n{String, Bool, Integer,
Integer}</td>
<td>The set of metrics to monitor during the monitoring period. Example values
are:\n{‘temp’, ‘true’, max_temp, min_temp}\n {‘clock’, ‘false’, max_clock,
min_clock}\n {‘mem_clock’, ‘true’, max_mem_clock, min_mem_clock}\n {‘fan’,
‘true’, max_fan, min_fan}\n {‘power’, ‘true’, max_power, min_power}\n The set of
upper bounds for each metric are specified as an integer. The units and values
for each metric are:\n temp - degrees Celsius\n clock - MHz \n mem_clock - MHz
\n fan - Integer between 0 and 255 \n power - Power in Watts</td></tr>
<tr><td>sample_interval</td><td>Integer</td>
<td>If this key is specified metrics will be sampled at the given rate. The
units for the sample_interval are milliseconds. The default value is 1000.
</td></tr>
<tr><td>log_interval</td><td>Integer</td>
<td>If this key is specified informational messages will be emitted at the given
interval, providing the current values of all parameters specified. This
parameter must be equal to or greater than the sample rate. If this value is not
specified, no logging will occur.</td></tr>
<tr><td>terminate</td><td>Bool</td> <td>If the terminate key is true the GM
monitor will terminate the RVS process when a bounds violation is encountered on
any of the metrics specified.</td></tr>
</table>

@subsection usg52 5.2 Output

Module specific output keys are described in the table below:
<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>metric_values</td><td>Time Series Collection of Result
Integers</td><td>A collection of integers containing the result values for each
of the metrics being monitored. </td></tr>
<tr><td>metric_violations</td><td>Collection of Result Integers </td><td>A
collection of integers containing the violation count for each of the metrics
being monitored. </td></tr>
<tr><td>metric_average</td><td>Collection of Result Integers </td><td></td></tr>
</table>

When monitoring is started for a target GPU, a result message is logged
with the following format:

    [RESULT][<timestamp>][<action name>] gm <gpu id> started

In addition, an informational message is provided for each for each metric
being monitored:

    [INFO ][<timestamp>][<action name>] gm <gpu id> monitoring < metric> bounds min:<min_metric> max: <max_metric>

During the monitoring informational output regarding the metrics of the GPU will
be sampled at every interval specified by the sample_rate key. If a bounding box
violation is discovered during a sampling interval, a warning message is
logged with the following format:

    [INFO ][<timestamp>][<action name>] gm <gpu id> < metric> bounds violation <metric value>

If the log_interval value is set an information message for each metric is
logged at every interval using the following format:

    [INFO ][<timestamp>][<action name>] gm <gpu id> < metric> <metric_value>

When monitoring is stopped for a target GPU, a result message is logged
with the following format:

    [RESULT][<timestamp>][<action name>] gm <gpu id> gm stopped

The following messages, reporting the number of metric violations that were
sampled over the duration of the monitoring and the average metric value is
reported:

    [RESULT][<timestamp>][<action name>] gm <gpu id> <metric> violations <metric_violations>
    [RESULT][<timestamp>][<action name>] gm <gpu id> <metric> average <metric_average>

@subsection usg53 5.3 Examples


@section usg6 6 PESM Module
The PCIe State Monitor (PESM) tool is used to actively monitor the PCIe
interconnect between the host platform and the GPU. The module registers
“listener” on a target GPUs PCIe interconnect, and log a message whenever it
detects a state change. The PESM is able to detect the following state changes:

1. PCIe link speed changes
2. GPU device power state changes

This module is intended to run concurrently with other actions, and provides a
‘start’ and ‘stop’ configuration key to start the monitoring and then stop it
after testing has completed. For information on GPU power state monitoring
please consult the 7.6. PCI Power Management Capability Structure, Gen 3 spec,
page 601, device states D0-D3. For information on link status changes please
consult the 7.8.8. Link Status Register (Offset 12h), Gen 3 spec, page 635.

Monitoring is performed by polling respective PCIe registers roughly every 1ms
(one millisecond).

@subsection usg61 6.1 Module Specific Keys
<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>monitor</td><td>Bool</td><td>This this key is set to true, the PESM
module will start monitoring on specified devices. If this key is set to false,
all other keys are ignored and monitoring will be stopped for all devices.</td>
</tr> </table>

@subsection usg62 6.2 Output

Module specific output keys are described in the table below:
<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>state</td><td>String</td><td>A string detailing the current power state
of the GPU or the speed of the PCIe link.</td></tr>
</table>

When monitoring is started for a target GPU, a result message is logged
with the following format:

    [RESULT][<timestamp>][<action name>] pesm <gpu id> started

When monitoring is stopped for a target GPU, a result message is logged
with the following format:

    [RESULT][<timestamp>][<action name>] pesm all stopped

When monitoring is enabled, any detected state changes in link speed or GPU
power state will generate the following informational messages:

    [INFO ][<timestamp>][<action name>] pesm <gpu id> power state change <state>
    [INFO ][<timestamp>][<action name>] pesm <gpu id> link speed change <state>

@subsection usg63 6.3 Examples

@section usg7 7 RCQT Module

This ‘module’ is actually a set of modules that target and qualify the
configuration of the platform. Many of the checks can be done manually using the
operating systems command line tools and general knowledge about ROCm’s
requirements. The purpose of the RCQT modules is to provide an extensible, OS
independent and scriptable interface capable for performing the configuration
checks required for ROCm support. The checks in this module do not target a
specific device (instead the underlying platform is targeted), and any device or
device id keys specified will be ignored. Iteration keys, i.e. count, wait and
duration, are also ignored.

@subsection usg71 7.1 Packaging Check
@subsubsection usg711 7.1.1 Packaging Check Specific Keys

Input keys are described in the table below:

<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>package</td><td>String</td>
<td>Specifies the package to check. This key is required.</td></tr>
<tr><td>version</td><td>String</td>
<td>This is an optional key specifying a package version. If it is provided, the
tool will check if the version is installed, failing otherwise. If it is not
provided any version matching the package name will result in success.
</td></tr>
</table>

@subsubsection usg712 7.1.2 Packaging Check Output Keys

Output keys are described in the table below:

<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>installed</td><td>Bool</td>
<td>If the test has passed, the output will be true. Otherwise it will be false.
</td></tr>
</table>

@section usg8 8 PEQT Module
PCI Express Qualification Tool module targets and qualifies the configuration of
the platforms PCIe connections to the GPUs. The purpose of the PEQT module is to
provide an extensible, OS independent and scriptable interface capable of
performing the PCIe interconnect configuration checks required for ROCm support
of GPUs. This information can be obtained through the sysfs PCIe interface or by
using the PCIe development libraries to extract values from various PCIe
control, status and capabilities registers. These registers are specified in the
PCI Express Base Specification, Revision 3. Iteration keys, i.e. count, wait and
duration will be ignored for actions using the PEQT module.

@subsection usg81 8.1 Module Specific Keys
Module specific output keys are described in the table below:
<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>capability</td><td>Collection of Structures with the
following format:\n{String,String}</td>
<td>The PCIe capability key contains a collection of structures that specify
which PCIe capability to check and the expected value of the capability. A check
structure must contain the PCIe capability value, but an expected value may be
omitted. The value of all valid capabilities that are a part of this collection
will be entered into the capability_value field. Possible capabilities, and
their value types are:\n\n
link_cap_max_speed\n
link_cap_max_width\n
link_stat_cur_speed\n
link_stat_neg_width\n
slot_pwr_limit_value\n
slot_physical_num\n
atomic_op_32_completer\n
atomic_op_64_completer\n
atomic_op_128_CAS_completer\n
atomic_op_routing\n
dev_serial_num\n
kernel_driver\n
pwr_base_pwr\n
pwr_rail_type\n
device_id\n
vendor_id\n\n

The expected value String is a regular expression that is used to check the
actual value of the capability.

</td></tr>
</table>

@subsection usg82 8.2 Output
Module specific output keys are described in the table below:
<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>capability_value</td><td>Collection of Strings</td>
<td>For each of the capabilities specified in the capability key, the actual
value of the capability will be returned, represented as a String.</td></tr>
<tr><td>pass</td><td>String</td> <td>'true' if all of the properties match the
values given, 'false' otherwise.</td></tr>
</table>

The qualification check queries the specified PCIe capabilities and
properties and checks that their actual values satisfy the regular expression
provided in the ‘expected value’ field for that capability. The pass output key
will be true and the test will pass if all of the properties match the values
given. After the check is finished, the following informational messages will be
generated:

    [INFO  ][<timestamp>][<action name>] peqt <capability> <capability_value>
    [RESULT][<timestamp>][<action name>] peqt <pass>

For details regarding each of the capabilities and current values consult the
chapters in the PCI Express Base Specification, Revision 3.

@subsection usg83 8.3 Examples

@section usg9 9 SMQT Module
The GPU SBIOS mapping qualification tool is designed to verify that a platform’s
SBIOS has satisfied the BAR mapping requirements for VDI and Radeon Instinct
products for ROCm support. These are the current BAR requirements:\n\n

BAR 1: GPU Frame Buffer BAR – In this example it happens to be 256M, but
typically this will be size of the GPU memory (typically 4GB+). This BAR has to
be placed < 2^40 to allow peer- to-peer access from other GFX8 AMD GPUs. For
GFX9 (Vega GPU) the BAR has to be placed < 2^44 to allow peer-to-peer access
from other GFX9 AMD GPUs.\n\n

BAR 2: Doorbell BAR – The size of the BAR is typically will be < 10MB (currently
fixed at 2MB) for this generation GPUs. This BAR has to be placed < 2^40 to
allow peer-to-peer access from other current generation AMD GPUs.\n\n
BAR 3: IO BAR - This is for legacy VGA and boot device support, but since this
the GPUs in this project are not VGA devices (headless), this is not a concern
even if the SBIOS does not setup.\n\n

BAR 4: MMIO BAR – This is required for the AMD Driver SW to access the
configuration registers. Since the reminder of the BAR available is only 1 DWORD
(32bit), this is placed < 4GB. This is fixed at 256KB.\n\n

BAR 5: Expansion ROM – This is required for the AMD Driver SW to access the
GPU’s video-BIOS. This is currently fixed at 128KB.\n\n

Refer to the ROCm Use of Advanced PCIe Features and Overview of How BAR Memory
is Used In ROCm Enabled System web page for more information about how BAR
memory is initialized by VDI and Radeon products. Iteration keys, i.e. count,
wait and duration will be ignored.

@subsection usg91 9.1 Module Specific Keys

Module specific output keys are described in the table below:
<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>bar1_req_size</td><td>Integer</td>
<td>This is an integer specifying the required size of the BAR1 frame buffer
region.</td></tr>
<tr><td>bar1_base_addr_min</td><td>Integer</td>
<td>This is an integer specifying the minimum value the BAR1 base address can
be.</td></tr>
<tr><td>bar1_base_addr_max</td><td>Integer</td>
<td>This is an integer specifying the maximum value the BAR1 base address can
be.</td></tr>
<tr><td>bar2_req_size</td><td>Integer</td>
<td>This is an integer specifying the required size of the BAR2 frame buffer
region.</td></tr>
<tr><td>bar2_base_addr_min</td><td>Integer</td>
<td>This is an integer specifying the minimum value the BAR2 base address can
be.</td></tr>
<tr><td>bar2_base_addr_max</td><td>Integer</td>
<td>This is an integer specifying the maximum value the BAR2 base address can
be.</td></tr>
<tr><td>bar4_req_size</td><td>Integer</td>
<td>This is an integer specifying the required size of the BAR4 frame buffer
region.</td></tr>
<tr><td>bar4_base_addr_min</td><td>Integer</td>
<td>This is an integer specifying the minimum value the BAR4 base address can
be.</td></tr>
<tr><td>bar4_base_addr_max</td><td>Integer</td>
<td>This is an integer specifying the maximum value the BAR4 base address can
be.</td></tr>
<tr><td>bar5_req_size</td><td>Integer</td>
<td>This is an integer specifying the required size of the BAR5 frame buffer
region.</td></tr>
</table>

@subsection usg92 9.2 Output
Module specific output keys are described in the table below:
<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>bar1_size</td><td>Integer</td><td>The actual size of BAR1.</td></tr>
<tr><td>bar1_base_addr</td><td>Integer</td><td>The actual base address of BAR1
memory.</td></tr>
<tr><td>bar2_size</td><td>Integer</td><td>The actual size of BAR2.</td></tr>
<tr><td>bar2_base_addr</td><td>Integer</td><td>The actual base address of BAR2
memory.</td></tr>
<tr><td>bar4_size</td><td>Integer</td><td>The actual size of BAR4.</td></tr>
<tr><td>bar4_base_addr</td><td>Integer</td><td>The actual base address of BAR4
memory.</td></tr>
<tr><td>bar5_size</td><td>Integer</td><td>The actual size of BAR5.</td></tr>
<tr><td>pass</td><td>String</td> <td>'true' if all of the properties match the
values given, 'false' otherwise.</td></tr>
</table>

The qualification check will query the specified bar properties and check that
they satisfy the give parameters. The pass output key will be true and the test
will pass if all of the BAR properties satisfy the constraints. After the check
is finished, the following informational messages will be generated:

    [INFO  ][<timestamp>][<action name>] smqt bar1_size <bar1_size>
    [INFO  ][<timestamp>][<action name>] smqt bar1_base_addr <bar1_base_addr>
    [INFO  ][<timestamp>][<action name>] smqt bar2_size <bar2_size>
    [INFO  ][<timestamp>][<action name>] smqt bar2_base_addr <bar2_base_addr>
    [INFO  ][<timestamp>][<action name>] smqt bar4_size <bar4_size>
    [INFO  ][<timestamp>][<action name>] smqt bar4_base_addr <bar4_base_addr>
    [INFO  ][<timestamp>][<action name>] smqt bar5_size <bar5_size>
    [RESULT][<timestamp>][<action name>] smqt <pass>


@subsection usg93 9.3 Examples

@section usg10 10 PQT Module
The P2P Qualification Tool is designed to provide the list of all GPUs that
support P2P and characterize the P2P links between peers. In addition to testing
for P2P compatibility, this test will perform a peer-to-peer throughput test
between all unique P2P pairs for performance evaluation. These are known as
device-to-device transfers, and can be either uni-directional or bi-directional.
The average bandwidth obtained is reported to help debug low bandwidth issues.

@subsection usg101 10.1 Module Specific Keys
<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>peers</td><td>Collection of Strings</td>
<td>This is a required key, and specifies the set of GPU(s) considered being
peers of the GPU specified in the action. If ‘all’ is specified, all other
GPU(s) on the system will be considered peers. Otherwise only the GPU ids
specified in the list will be considered.</td></tr>
<tr><td>peer_deviceid</td><td>Integer</td>
<td>This is an optional parameter, but if specified it restricts the peers list
to a specific device type corresponding to the deviceid.</td></tr>
<tr><td>test_bandwidth</td><td>Bool</td>
<td>If this key is set to true the P2P bandwidth benchmark will run if a pair of
devices pass the P2P check.</td></tr>
<tr><td>bidirectional</td><td>Bool</td>
<td>This option is only used if test_bandwidth key is true. This specifies the
type of transfer to run:\n
true – Do a bidirectional transfer test\n
false – Do a unidirectional transfer test
from the agent to its peers.
</td></tr>
<tr><td>parallel</td><td>Bool</td>
<td>This option is only used if the test_bandwith
key is true.\n
true – Run transfer testing to all peers
in parallel.\n
false – Run transfer testing to a single
peer at a time.
</td></tr>
<tr><td>duration</td><td>Integer</td>
<td>This option is only used if test_bandwidth is true. This key specifies the
duration a transfer test should run, given in milliseconds. If this key is not
specified, the default value is 10000 (10 seconds).
</td></tr>
<tr><td>log_interval</td><td>Integer</td>
<td>This option is only used if test_bandwidth is true. This is a positive
integer, given in milliseconds, that specifies an interval over which the moving
average of the bandwidth will be calculated and logged. The default value is
1000 (1 second). It must be smaller than the duration key.</td></tr>
</table>

@subsection usg102 10.2 Output

Module specific output keys are described in the table below:
<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>p2p_result</td><td>Collection of Result Bools</td>
<td>Indicates if the gpu and the specified peer have P2P capabilities. If this
quantity is true, the GPU pair tested has p2p capabilities. If false, they are
not peers.</td></tr>
<tr><td>interval_bandwidth</td><td>Collection of Time Series Floats</td>
<td>The average bandwidth of a p2p transfer, during the log_interval time
period. </td></tr>
<tr><td>bandwidth</td><td>Collection of Floats</td>
<td>The average bandwidth of a p2p transfer, averaged over the entire test
duration of the interval.</td></tr>
</table>

If the value of test_bandwidth key is false, the tool will only try to determine
if the GPU(s) in the peers key are P2P to the action’s GPU. In this case the
bidirectional and log_interval values will be ignored, if they are specified. If
a gpu is a P2P peer to the device the test will pass, otherwise it will fail. A
message indicating the result will be provided for each GPUs specified. It will
have the following format:

    [RESULT][<timestamp>][<action name>] p2p <gpu id> <peer gpu id> <p2p_result>

If the value of test_bandwidth is true bandwidth testing between the device and
each of its peers will take place in parallel or in sequence, depending on the
value of the parallel flag. During the duration of bandwidth benchmarking,
informational output providing the moving average of the transfer’s bandwidth
will be calculated and logged at every time increment specified by the
log_interval parameter. The messages will have the following output:

    [INFO  ][<timestamp>][<action name>] p2p-bandwidth <gpu id> <peer gpu id> bidirectional: <bidirectional> <interval_bandwidth >

At the end of the test the average bytes/second will be calculated over the
entire test duration, and will be logged as a result:

    [RESULT][<timestamp>][<action name>] p2p-bandwidth <gpu id> <peer gpu id> bidirectional: <bidirectional> <bandwidth > <duration>

@subsection usg103 10.3 Examples

@section usg11 11 PEBB Module
The PCIe Bandwidth Benchmark attempts to saturate the PCIe bus with DMA
transfers between system memory and a target GPU card’s memory. These are known
as host-to-device or device- to-host transfers, and can be either unidirectional
or bidirectional transfers. The maximum bandwidth obtained is reported.

@subsection usg111 11.1 Module Specific Keys

<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>host_to_device</td><td>Bool</td>
<td>This key indicates if host to device transfers
will be considered. The default value is true.</td></tr>
<tr><td>device_to_host</td><td>Bool</td>
<td>This key indicates if device to host transfers
will be considered. The default value is true.
</td></tr>
<tr><td>log_interval</td><td>Integer</td>
<td>This is a positive integer, given in milliseconds, that specifies an
interval over which the moving average of the bandwidth will be calculated and
logged.</td></tr>
</table>

@subsection usg112 11.2 Output

Module specific output keys are described in the table below:
<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>interval_bandwidth</td><td>Collection of Time Series Floats</td>
<td>The average bandwidth of a transfer, during the log_interval time
period. </td></tr>
<tr><td>bandwidth</td><td>Collection of Floats</td>
<td>The average bandwidth of a transfer, averaged over the entire test
duration of the interval.</td></tr>
</table>

During the execution of the benchmark, informational output providing the moving
average of the bandwidth of the transfer will be calculated and logged. This
interval is provided by the log_interval parameter and will have the following
output format:

    [INFO ][<timestamp>][<action name>] pcie-bandwidth <gpu id> h2d: <host_to_device> d2h: <device_to_host> <interval_bandwidth >

At the end of the test the average bytes/second will be calculated over the
entire test duration, and will be logged as a result:

    [RESULT][<timestamp>][<action name>] pcie-bandwidth <gpu id> h2d: <host_to_device> d2h: <device_to_host> < bandwidth > <duration>


@subsection usg113 11.3 Examples

@section usg12 12 GST Module
The GPU Stress Test modules purpose is to bring the CUs of the specified GPU(s)
to a target performance level in gigaflops by doing large matrix multiplications
using SGEMM/DGEMM (Single/Double-precision General Matrix Multiplication)
available in a library like rocBlas. The GPU stress module may be configured so
it does not copy the source arrays to the GPU before every matrix
multiplication. This allows the GPU performance to not be capped by device to
host bandwidth transfers. The module calculates how many matrix operations per
second are necessary to achieve the configured performance target and fails if
it cannot achieve that target. \n\n

This module should be used in conjunction with the GPU Monitor, to watch for
thermal, power and related anomalies while the target GPU(s) are under realistic
load conditions. By setting the appropriate parameters a user can ensure that
all GPUs in a node or cluster reach desired performance levels. Further analysis
of the generated stats can also show variations in the required power, clocks or
temperatures to reach these targets, and thus highlight GPUs or nodes that are
operating less efficiently.

@subsection usg121 12.1 Module Specific Keys

Module specific keys are described in the table below:

<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>target_stress</td><td>Float</td>
<td>The maximum relative performance the GPU will attempt to achieve in
gigaflops. This parameter is required.</td></tr>
<tr><td>copy_matrix</td><td>Bool</td>
<td>This parameter indicates if each operation should copy the matrix data to
the GPU before executing. The default value is true.</td></tr>
<tr><td>ramp_interval</td><td>Integer</td>
<td>This is an time interval, specified in milliseconds, given to the test to
reach the given target_stress gigaflops. The default value is 5000 (5 seconds).
This time is counted against the duration of the test. If the target gflops, or
stress, is not achieved in this time frame, the test will fail. If the target
stress (gflops) is achieved the test will attempt to run for the rest of the
duration specified by the action, sustaining the stress load during that
time.</td></tr>
<tr><td>tolerance</td><td>Float</td>
<td>A value indicating how much the target_stress can fluctuate after the ramp
period for the test to succeed. The default value is 0.1 or 10%.</td></tr>
<tr><td>max_violations</td><td>Integer</td>
<td>The number of tolerance violations that can occur after the ramp_interval
for the test to still pass. The default value is 0.</td></tr>
<tr><td>log_interval</td><td>Integer</td>
<td>This is a positive integer, given in milliseconds, that specifies an
interval over which the moving average of the bandwidth will be calculated and
logged.</td></tr>
<tr><td>matrix_size</td><td>Integer</td>
<td>Size of the matrices of the SGEMM operations. The default value is
5760.</td></tr>
</table>

@subsection usg122 12.2 Output

Module specific output keys are described in the table below:

<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>target_stress</td><td>Time Series Floats</td>
<td>The average gflops over the last log interval.</td></tr>
<tr><td>max_gflops</td><td>Float</td>
<td>The maximum sustained performance obtained by the GPU during the
test.</td></tr>
<tr><td>stress_violations</td><td>Integer</td>
<td>The number of gflops readings that violated the tolerance of the test after
the ramp interval.</td></tr>
<tr><td>flops_per_op</td><td>Integer</td>
<td>Flops (floating point operations) per operation queued to the GPU queue.
One operation is one call to SGEMM/DGEMM.</td></tr>
<tr><td>bytes_copied_per_op</td><td>Integer</td>
<td>Calculated number of ops/second necessary to achieve target
gigaflops.</td></tr>
<tr><td>try_ops_per_sec</td><td>Float</td>
<td>Calculated number of ops/second necessary to achieve target
gigaflops.</td></tr>
<tr><td>pass</td><td>Bool</td>
<td>'true' if the GPU achieves its desired sustained performance
level.</td></tr>
</table>

An informational message indicating will be emitted when the test starts
execution:

    [INFO ][<timestamp>][<action name>] gst <gpu id> start <target_stress> copy matrix: <copy_matrix>


During the execution of the test, informational output providing the moving
average the GPU(s) gflops will be logged at each log_interval:

    [INFO ][<timestamp>][<action name>] gst Gflops: <interval_gflops>

When the target gflops is achieved, the following message will be logged:

    [INFO ][<timestamp>][<action name>] gst <gpu id> target achieved <target_stress>

If the target gflops, or stress, is not achieved in the “ramp_interval”
provided, the test will terminate and the following message will be logged:

    [INFO ][<timestamp>][<action name>] gst <gpu id> ramp time exceeded <ramp_time>

In this case the test will fail.\n

If the target stress (gflops) is achieved the test will attempt to run for the
rest of the duration specified by the action, sustaining the stress load during
that time. If the stress level violates the bounds set by the tolerance level
during that time a violation message will be logged:

    [INFO ][<timestamp>][<action name>] gst <gpu id> stress violation <interval_gflops>

When the test completes, the following result message will be printed:

    [RESULT][<timestamp>][<action name>] gst <gpu id> Gflop: < max_gflops> flops_per_op:<flops_per_op> bytes_copied_per_op: <bytes_copied_per_op> try_ops_per_sec: <try_ops_per_sec> pass: <pass>

The test will pass if the target_stress is reached before the end of the
ramp_interval and the stress_violations value is less than the given
max_violations value. Otherwise, the test will fail.

@subsection usg123 12.3 Examples

When running the __GST__ module, users should provide at least an action name,
the module name (gst), a list of GPU IDs, the test duration and a target stress
value (gigaflops). Thus, the most basic configuration file looks like this:

    actions:
    - name: action_gst_1
      module: gst
      device: all
      target_stress: 3500
      duration: 8000

For the above configuration file, all the missing configuration keys will have
their default
values (e.g.: __copy_matrix=true__, __matrix_size=5760__ etc.). For more
information about the default
values please consult the dedicated sections (__3.3 Common Configuration Keys__
and __5.1 Configuration keys__).

When the __RVS__ tool runs against such a configuration file, it will do the
following:
  - run the stress test on all available (and compatible) AMD GPUs, one after
the other
  - log a start message containing the GPU ID, the __target_stress__ and the
value of the __copy_matrix__<br />
e.g.: __[INFO  ] [164337.932824] action_gst_1 gst 50599 start 3500.000000 copy
matrix:true__
  - emit, each __log_interval__ (e.g.: 1000ms), a message containing the
gigaflops value that the current GPU achieved<br />
e.g.: __[INFO  ] [164355.111207] action_gst_1 gst 33367 Gflops 3535.670231__
  - log a message as soon as the current GPU reaches the given
__target_stress__<br />
e.g.: __[INFO  ] [164350.804843] action_gst_1 gst 33367 target achieved
3500.000000__
  - log a __ramp time exceeded__ message if the GPU was not able to reach the
__target_stress__ in the __ramp_interval__ time frame (e.g.: 5000). In such a
case, the test will also terminate<br />
e.g.: __[INFO  ] [164013.788870] action_gst_1 gst 3254 ramp time exceeded 5000__
  - log the test result, when the stress test completes. The message contains
the test's overall result and some other statistics according to __5.2 Output
keys__<br />
e.g.: __[RESULT] [164355.647523] action_gst_1 gst 33367 Gflop: 4066.020766
flops_per_op: 382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec:
9.157367 pass: TRUE__
  - log a __stress violation__ message when the current gigaflops (for the last
__log_interval__, e.g.; 1000ms) violates the bounds set by the __tolerance__
configuration key (e.g.: 0.1). Please note that this message is not logged
during the __ramp_interval__ time frame<br />
e.g.: __[INFO  ] [164013.788870] action_gst_1 gst 3254 stress violation 2500__


If a mandatory configuration key is missing, the __RVS__ tool will log an error
message and terminate the executation of the current module. For example, the
following configuration file will cause the __RVS__ to terminate with the
following error message:<br /> __RVS-GST: action: action_gst_1  key
'target_stress' was not found__

    actions:
    - name: action_gst_1
      module: gst
      device: all
      duration: 8000

A more complex configuration file looks like this:

    actions:
    - name: action_1
      device: 50599 33367
      module: gst
      parallel: false
      count: 12
      wait: 100
      duration: 7000
      ramp_interval: 3000
      log_interval: 1000
      max_violations: 2
      copy_matrix: false
      target_stress: 5000
      tolerance: 0.07
      matrix_size: 5760

For this configuration file, the RVS tool:
  - will run the stress test only for the GPUs having the ID 50599 or 33367. To
get all the available GPU IDs, run __RVS__ tool with __-g__ option
  - will run the test on the selected GPUs, one after the other
  - will run each test, 12 times
  - will only copy the matrices to the GPUs at the beginning of the test
  - will wait 100ms before each test execution
  - will try to reach 5000 gflops in maximum 3000ms
  - if __target_stress__ (5000) is achieved in the __ramp_interval__ (3000 ms)
it will attempt to run the test for the rest of the duration, sustaining the
stress load during that time
  - will allow a 7% __target_stress__ __tolerance__ (each __target_stress__
violation will generate a __stress violation__ message as shown in the first
example)
  - will allow only 2 __target_stress__ violations. Exceeding the
__max_violations__ will not terminate the test, but the __RVS__ will mark the
test result as "fail".

The output for such a configuration key may look like this:

__[INFO  ] [172061.758830] action_1 gst 50599 start 5000.000000 copy
matrix:false__<br />
__[INFO  ] [172063.547668] action_1 gst 50599 Gflops 6471.614725__<br />
__[INFO  ] [172064.577715] action_1 gst 50599 target achieved 5000.000000__<br
/>
__[INFO  ] [172065.609224] action_1 gst 50599 Gflops 5189.993529__<br />
__[INFO  ] [172066.634360] action_1 gst 50599 Gflops 5220.373979__<br />
__[INFO  ] [172067.659262] action_1 gst 50599 Gflops 5225.472000__<br />
__[INFO  ] [172068.694305] action_1 gst 50599 Gflops 5169.935583__<br />
__[RESULT] [172069.573967] action_1 gst 50599 Gflop: 6471.614725 flops_per_op:
382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass:
TRUE__<br />
__[INFO  ] [172069.574369] action_1 gst 33367 start 5000.000000 copy
matrix:false__<br />
__[INFO  ] [172071.409483] action_1 gst 33367 Gflops 6558.348080__<br />
__[INFO  ] [172072.438104] action_1 gst 33367 target achieved 5000.000000__<br
/>
__[INFO  ] [172073.465033] action_1 gst 33367 Gflops 5215.285895__<br />
__[INFO  ] [172074.501571] action_1 gst 33367 Gflops 5164.945297__<br />
__[INFO  ] [172075.529468] action_1 gst 33367 Gflops 5210.207720__<br />
__[INFO  ] [172076.558102] action_1 gst 33367 Gflops 5205.139424__<br />
__[RESULT] [172077.448182] action_1 gst 33367 Gflop: 6558.348080 flops_per_op:
382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass:
TRUE__<br />

When setting the __parallel__ to false, the __RVS__ will run the stress tests on
all selected GPUs in parallel and the output may look like this:

__[INFO  ] [173381.407428] action_1 gst 50599 start 5000.000000 copy
matrix:false__<br />
__[INFO  ] [173381.407744] action_1 gst 33367 start 5000.000000 copy
matrix:false__<br />
__[INFO  ] [173383.245771] action_1 gst 33367 Gflops 6558.348080__<br />
__[INFO  ] [173383.256935] action_1 gst 50599 Gflops 6484.532120__<br />
__[INFO  ] [173384.274202] action_1 gst 33367 target achieved 5000.000000__<br
/>
__[INFO  ] [173384.286014] action_1 gst 50599 target achieved 5000.000000__<br
/>
__[INFO  ] [173385.301038] action_1 gst 33367 Gflops 5215.285895__<br />
__[INFO  ] [173385.315794] action_1 gst 50599 Gflops 5200.080980__<br />
__[INFO  ] [173386.337638] action_1 gst 33367 Gflops 5164.945297__<br />
__[INFO  ] [173386.353274] action_1 gst 50599 Gflops 5159.964636__<br />
__[INFO  ] [173387.365494] action_1 gst 33367 Gflops 5210.207720__<br />
__[INFO  ] [173387.383437] action_1 gst 50599 Gflops 5195.032357__<br />
__[INFO  ] [173388.401250] action_1 gst 33367 Gflops 5169.935583__<br />
__[INFO  ] [173388.421599] action_1 gst 50599 Gflops 5154.993572__<br />
__[RESULT] [173389.282710] action_1 gst 33367 Gflop: 6558.348080 flops_per_op:
382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass:
TRUE__<br />
__[RESULT] [173389.305479] action_1 gst 50599 Gflop: 6484.532120 flops_per_op:
382.205952x1e9 bytes_copied_per_op: 398131200 try_ops_per_sec: 13.081952 pass:
TRUE__<br />

It is important that all the configuration keys will be adjusted/fine-tuned
according to the actual GPUs and HW platform capabilities. For example, a matrix
size of 5760 should fit the VEGA 10 GPUs while 8640 should work with the VEGA 20
GPUs.

@section usg13 13 IET Module

The Input EDPp Test can be used to characterize the peak power capabilities of a
GPU to different levels of use. This tool can leverage the functionality of the
GST to drive the compute load on the GPU, but the test will use different
configuration and output keys and should focus on driving power usage rather
than calculating compute load. The purpose of the IET module is to bring the
GPU(s) to a preconfigured power level in watts by gradually increasing the
compute load on the GPUs until the desired power level is achieved. This
verifies that the GPUs can sustain a power level for a reasonable amount of time
without problems like thermal violations arising.\n

This module should be used in conjunction with the GPU Monitor, to watch for
thermal, power and related anomalies while the target GPU(s) are under realistic
load conditions. By setting the appropriate parameters a user can ensure that
all GPUs in a node or cluster reach desired performance levels. Further analysis
of the generated stats can also show variations in the required power, clocks or
temperatures to reach these targets, and thus highlight GPUs or nodes that are
operating less efficiently.

@subsection usg131 13.1 Module Specific Keys

Module specific keys are described in the table below:

<table>
<tr><th>Config Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>target_power</td><td>Float</td>
<td>This is a floating point value specifying the target sustained power level
for the test.</td></tr>
<tr><td>ramp_interval</td><td>Integer</td>
<td>This is an time interval, specified in milliseconds, given to the test to
determine the compute load that will sustain the target power. The default value
is 5000 (5 seconds). This time is counted against the duration of the test.
</td></tr>
<tr><td>tolerance</td><td>Float</td>
<td>A value indicating how much the target_power can fluctuate after the ramp
period for the test to succeed. The default value is 0.1 or 10%.
</td></tr>
<tr><td>max_violations</td><td>Integer</td>
<td>The number of tolerance violations that can occur after the ramp_interval
for the test to still pass. The default value is 0.</td></tr>
<tr><td>sample_interval</td><td>Integer</td>
<td>The sampling rate for target_power values given in milliseconds. The default
value is 100 (.1 seconds).
</td></tr>
<tr><td>log_interval</td><td>Integer</td>
<td>This is a positive integer, given in milliseconds, that specifies an
interval over which the moving average of the bandwidth will be calculated and
logged.</td></tr>
</table>


@subsection usg132 13.2 Output

Module specific output keys are described in the table below:

<table>
<tr><th>Output Key</th> <th>Type</th><th> Description</th></tr>
<tr><td>current_power</td><td>Time Series Floats</td>
<td>The current measured power of the GPU.</td></tr>
<tr><td>power_violations</td><td>Integer</td>
<td>The number of power reading that violated the tolerance of the test after
the ramp interval.
</td></tr>
<tr><td>pass</td><td>Bool</td>
<td>'true' if the GPU achieves its desired sustained power level in the ramp
interval.</td></tr>
</table>

@subsection usg133 13.3 Examples

