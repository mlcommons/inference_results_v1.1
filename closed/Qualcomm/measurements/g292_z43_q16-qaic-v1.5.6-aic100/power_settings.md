# Boot/BIOS Firmware Settings

## AMD CBS

### NBIO Common Options
#### SMU Common Options
##### Determinism Slider = Power
##### cTDP = 280 W
##### Package Power Limit = 280 W
##### DF Cstates = Disabled
##### Fixed SOC Pstate = P0

### DF Common Options
#### Memory Addressing
##### NUMA nodes per socket = NPS1
##### ACPI SRAT L3 Cche As NUMA Domain = Disabled

### CPU Common Options
#### L1 Stream HW Prefetcher = Enable
#### L1 Stride Prefetcher = Auto
#### L1 Region Prefetcher = Auto
#### L2 Stream HW Prefetcher = Enable
#### L2 Up/Down Prefetcher = Auto

# Management Firmware Settings

Out-of-the-box.

# Power Management Settings

## Fan Settings (10800 RPM)

<pre>
<b>&dollar;</b> ipmitool -I lanplus -U admin -P password -H aus655-perf-g292-3-bmc.qualcomm.com raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>150</b> 0xFF
 0a 3c 00
<b>&dollar;</b> ipmitool -I lanplus -U admin -P password -H aus655-perf-g292-3-bmc.qualcomm.com sensor get SYS_FAN1
Locating sensor record...
Sensor ID              : SYS_FAN1 (0xa2)
 Entity ID             : 29.3
 Sensor Type (Threshold)  : Fan
 Sensor Reading        : <b>10800</b> (+/- 0) RPM
 Status                : ok
 Lower Non-Recoverable : 0.000
 Lower Critical        : 1200.000
 Lower Non-Critical    : 1500.000
 Upper Non-Critical    : 38250.000
 Upper Critical        : 38250.000
 Upper Non-Recoverable : 38250.000
 Positive Hysteresis   : Unspecified
 Negative Hysteresis   : 150.000
 Assertion Events      :
 Assertions Enabled    : lnc- lnc+ lcr- lcr+ lnr- lnr+ unc- unc+ ucr- ucr+ unr- unr+
 Deassertions Enabled  : lnc- lnc+ lcr- lcr+ lnr- lnr+ unc- unc+ ucr- ucr+ unr- unr+
</pre>
