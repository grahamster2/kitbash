# Registers the Kitbash server to start at boot, and opens its port to
# Tailscale only. Run elevated.
#
# Scheduled Task rather than a real Windows service: a service needs a wrapper
# (NSSM/WinSW) to supervise a Python process, and a task at startup running as
# SYSTEM gets us boot survival and auto-restart with nothing extra installed.

$ErrorActionPreference = "Continue"

$TaskName = "KitbashServer"
$RunPs1   = "D:\kitbash\server\run.ps1"

if (-not (Test-Path $RunPs1)) { Write-Output "MISSING: $RunPs1"; exit 1 }

# --- scheduled task ---------------------------------------------------------
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$RunPs1`""

$trigger   = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

# RestartCount/Interval cover a crash; ExecutionTimeLimit 0 means "never kill it".
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings | Out-Null

Write-Output "registered scheduled task: $TaskName"

# --- firewall ---------------------------------------------------------------
# 100.64.0.0/10 is the CGNAT range Tailscale allocates from. Scoping to it means
# the inference server is not reachable from the LAN, a coffee-shop network, or
# the internet — only from devices in the tailnet. Tailscale uses these
# addresses even when the underlying path is a direct LAN connection.
Remove-NetFirewallRule -DisplayName "Kitbash server (Tailscale)" -ErrorAction SilentlyContinue

New-NetFirewallRule -DisplayName "Kitbash server (Tailscale)" `
    -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8188 `
    -RemoteAddress 100.64.0.0/10 -Profile Any | Out-Null

Write-Output "firewall rule added: TCP 8188 from 100.64.0.0/10 only"

Start-ScheduledTask -TaskName $TaskName
Write-Output "task started"
