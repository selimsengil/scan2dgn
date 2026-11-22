# Overnight: wait for the Qwen run to finish, build RAPOR.html, copy it to the Desktop, then force a shutdown.
# Runs on its own (started detached), so nothing else has to stay awake.
param([string]$Run = "kat_test\qwen_019_p3", [string]$Floor = "019", [int]$MaxHours = 6, [switch]$NoShutdown)
$ErrorActionPreference = "Continue"
Set-Location C:\Users\Selim\Documents\Proje\oda_kural
$log = Join-Path $Run "gece_log.txt"
function Say($m) { Add-Content -Encoding utf8 $log ("{0:HH:mm:ss} {1}" -f (Get-Date), $m) }
$name = Split-Path $Run -Leaf
Say "waiting for the run $name"
$deadline = (Get-Date).AddHours($MaxHours)
while ($true) {
    $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*kat_agent_api.py*$name*" }
    if (-not $p) { Say "run finished"; break }
    if ((Get-Date) -gt $deadline) { Say "time limit reached: stopping the run"; $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; Start-Sleep 5; break }
    Start-Sleep 30
}
& C:\Users\Selim\Documents\Proje\.venv\Scripts\python.exe rapor_gece.py $Run $Floor *>> $log
$rap = Join-Path $Run "RAPOR.html"
if (Test-Path $rap) {
    $desk = [Environment]::GetFolderPath("Desktop")
    Copy-Item $rap (Join-Path $desk "Qwen_kat_${Floor}_raporu.html") -Force
    Say "report copied to the Desktop"
} else { Say "NO REPORT written" }
if ($NoShutdown) { Say "test mode: no shutdown"; exit }
Say "shutting down"
shutdown.exe /s /f /t 60 /c "Qwen deneyi bitti, rapor masaustunde. Kapaniyor."
