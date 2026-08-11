Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class SleepUtil {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
    
    public const uint ES_CONTINUOUS = 0x80000000;
    public const uint ES_SYSTEM_REQUIRED = 0x00000001;
    public const uint ES_DISPLAY_REQUIRED = 0x00000002;
}
"@
# Prevent sleep and keep display on
[SleepUtil]::SetThreadExecutionState([SleepUtil]::ES_CONTINUOUS -bor [SleepUtil]::ES_SYSTEM_REQUIRED -bor [SleepUtil]::ES_DISPLAY_REQUIRED)

Write-Host "System is now caffeinated. Press Ctrl+C to exit and restore normal sleep behavior."

try {
    while ($true) {
        Start-Sleep -Seconds 60
    }
}
finally {
    # Restore normal sleep state on exit
    [SleepUtil]::SetThreadExecutionState([SleepUtil]::ES_CONTINUOUS)
    Write-Host "Sleep behavior restored."
}
