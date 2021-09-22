# Boot/BIOS Firmware Settings

Out-of-the-box.

# Management Firmware Settings

Out-of-the-box.

# Power Management Settings

## Fan Settings (8100 RPM)

<pre>
<b>&dollar;</b> ipmitool -I lanplus -U admin -P password -H aus655-pci-blanco-bmc.qualcomm.com raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>100</b> 0xFF
 0a 3c 00
<b>&dollar;</b> ipmitool -I lanplus -U admin -P password -H aus655-pci-blanco-bmc.qualcomm.com sensor get BPB_FAN_1A
Locating sensor record...
Sensor ID              : BPB_FAN_1A (0xa0)
 Entity ID             : 29.1
 Sensor Type (Threshold)  : Fan
 Sensor Reading        : <b>8100</b> (+/- 0) RPM
 Status                : ok
 Lower Non-Recoverable : na
 Lower Critical        : 1200.000
 Lower Non-Critical    : 1500.000
 Upper Non-Critical    : na
 Upper Critical        : na
 Upper Non-Recoverable : na
 Positive Hysteresis   : Unspecified
 Negative Hysteresis   : 150.000
 Assertion Events      :
 Assertions Enabled    : lnc- lcr-
 Deassertions Enabled  : lnc- lcr-
</pre>
