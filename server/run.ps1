# Launches the Kitbash GPU server.
#
# Every environment variable the server needs is set here rather than inherited.
# The scheduled task runs as SYSTEM, which does not see the per-user variables
# that `setx` wrote — a service that works interactively and dies at boot is
# almost always this.

$ErrorActionPreference = "Continue"

$Root       = "D:\kitbash"
$Venv       = "$Root\server\.venv\Scripts\python.exe"
$ServerDir  = "$Root\server"
$LogDir     = "D:\kitbash-logs"

$env:HF_HOME            = "D:\hf-cache"
$env:UV_CACHE_DIR       = "D:\uv-cache"
$env:HY3DGEN_MODELS     = "D:\hy3dgen-models"   # Hunyuan3D ignores HF_HOME
$env:KITBASH_HY3D_REPO  = "D:\models\Hunyuan3D-2.1"
$env:KITBASH_OUT_DIR    = "D:\kitbash-out"
$env:PYTHONUNBUFFERED   = "1"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$log = Join-Path $LogDir "server.log"

"[$(Get-Date -Format o)] starting kitbash server" | Out-File -Append $log

Set-Location $ServerDir
& $Venv -m uvicorn app:api --host 0.0.0.0 --port 8188 --log-level info *>&1 |
    Out-File -Append $log

"[$(Get-Date -Format o)] server exited with code $LASTEXITCODE" | Out-File -Append $log
