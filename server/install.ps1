# server/install.ps1
# Registers the "PsyFit server" scheduled task, which runs server/watchdog.py:
#   * at logon (so the server comes back after reboots, including Windows Update ones)
#   * every 5 minutes (re-launches the watchdog if it ever dies; a second copy exits at once)
# Runs only while you are signed in, as you, without admin rights: it needs your
# Google Drive (G:) mount for the lab-drive sync. Lock the screen rather than signing out.
#
# Usage:  powershell -ExecutionPolicy Bypass -File server\install.ps1
# Undo:   powershell -ExecutionPolicy Bypass -File server\uninstall.ps1

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$pythonw = Join-Path $repo 'venv\Scripts\pythonw.exe'
$watchdog = Join-Path $PSScriptRoot 'watchdog.py'
$task = 'PsyFit server'
$user = "$env:USERDOMAIN\$env:USERNAME"

if (-not (Test-Path $pythonw)) { throw "No venv found at $pythonw" }

$action = New-ScheduledTaskAction -Execute $pythonw -Argument "`"$watchdog`"" -WorkingDirectory $repo
$repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 5)
$logon = New-ScheduledTaskTrigger -AtLogOn -User $user
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited

try {
    Register-ScheduledTask -TaskName $task -Action $action -Trigger @($logon, $repeat) `
        -Settings $settings -Principal $principal -Force | Out-Null
    Write-Host "Registered '$task' (at logon + every 5 min)."
} catch {
    # A logon trigger can need admin rights; the 5-minute trigger alone still brings the
    # server up within 5 minutes of signing in.
    Register-ScheduledTask -TaskName $task -Action $action -Trigger $repeat `
        -Settings $settings -Principal $principal -Force | Out-Null
    Write-Host "Registered '$task' (every 5 min; logon trigger needs admin, skipped)."
}

Start-ScheduledTask -TaskName $task
$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {
        $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } |
    Select-Object -First 1).IPAddress
Write-Host "Started. Give it ~30 s, then open:"
Write-Host "  on this PC:      http://localhost:8501"
Write-Host "  from other PCs:  http://${env:COMPUTERNAME}:8501   or   http://${ip}:8501"
