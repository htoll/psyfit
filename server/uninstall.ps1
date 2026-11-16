# server/uninstall.ps1
# Stops the PsyFit server and removes its scheduled task. Usage data in server/data/
# is kept; delete that folder too if you want it gone.

$task = 'PsyFit server'
$data = Join-Path $PSScriptRoot 'data'

Unregister-ScheduledTask -TaskName $task -Confirm:$false -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $data | Out-Null
Set-Content (Join-Path $data 'stop.request') 'stop'    # watchdog stops the app and exits
Write-Host "Removed '$task'. The server stops within ~15 s."
