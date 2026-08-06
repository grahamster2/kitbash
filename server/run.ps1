# Launches the Kitbash GPU server.
#
# Every environment variable the server needs is set here rather than inherited.
# The scheduled task runs as SYSTEM, which does not see the per-user variables
# that `setx` wrote — a service that works interactively and dies at boot is
# almost always this.

$ErrorActionPreference = "Continue"

# Paths default to the reference layout but every one is overridable, so this
# works from wherever the repo actually lives. $ServerDir is derived from this
# script's own location rather than assumed.
$ServerDir  = $PSScriptRoot
$Venv       = if ($env:KITBASH_PYTHON) { $env:KITBASH_PYTHON } else { "$ServerDir\.venv\Scripts\python.exe" }
$LogDir     = if ($env:KITBASH_LOG_DIR) { $env:KITBASH_LOG_DIR } else { "D:\kitbash-logs" }
$Port       = if ($env:KITBASH_PORT) { $env:KITBASH_PORT } else { "8188" }

function Default-Env($name, $value) {
    if (-not [Environment]::GetEnvironmentVariable($name)) {
        [Environment]::SetEnvironmentVariable($name, $value)
    }
}

Default-Env "HF_HOME"           "D:\hf-cache"
Default-Env "UV_CACHE_DIR"      "D:\uv-cache"
Default-Env "HY3DGEN_MODELS"    "D:\hy3dgen-models"   # Hunyuan3D ignores HF_HOME
Default-Env "KITBASH_HY3D_REPO" "D:\models\Hunyuan3D-2.1"
Default-Env "KITBASH_OUT_DIR"   "D:\kitbash-out"
$env:PYTHONUNBUFFERED = "1"

if (-not (Test-Path $Venv)) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    "[$(Get-Date -Format o)] no python at $Venv" | Out-File -Append (Join-Path $LogDir "server.log")
    exit 1
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$log = Join-Path $LogDir "server.log"

"[$(Get-Date -Format o)] starting kitbash server" | Out-File -Append $log

Set-Location $ServerDir
& $Venv -m uvicorn app:api --host 0.0.0.0 --port $Port --log-level info *>&1 |
    Out-File -Append $log

"[$(Get-Date -Format o)] server exited with code $LASTEXITCODE" | Out-File -Append $log
