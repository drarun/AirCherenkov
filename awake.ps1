Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class SleepUtil {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
    
    public const uint ES_CONTINUOUS        = 0x80000000;
    public const uint ES_SYSTEM_REQUIRED   = 0x00000001;
    public const uint ES_DISPLAY_REQUIRED  = 0x00000002;
    public const uint ES_AWAYMODE_REQUIRED = 0x00000040;
}
"@

$wscript = New-Object -ComObject Wscript.Shell

Write-Host "System is now caffeinated with periodic execution state refresh and virtual keep-alive."
Write-Host "Press Ctrl+C to exit and restore normal sleep behavior."

try {
    while ($true) {
        # 1. Periodically refresh execution state with System + Display + AwayMode
        [SleepUtil]::SetThreadExecutionState(
            [SleepUtil]::ES_CONTINUOUS -bor 
            [SleepUtil]::ES_SYSTEM_REQUIRED -bor 
            [SleepUtil]::ES_DISPLAY_REQUIRED -bor 
            [SleepUtil]::ES_AWAYMODE_REQUIRED
        )
        
        # 2. Send F15 keypress to reset Windows physical user input idle timer
        try {
            $wscript.SendKeys("{F15}")
        } catch {
            # Ignore if SendKeys fails in background session
        }
        
        Start-Sleep -Seconds 30
    }
}
finally {
    # Restore normal sleep state on exit
    [SleepUtil]::SetThreadExecutionState([SleepUtil]::ES_CONTINUOUS)
    Write-Host "Sleep behavior restored."
}
