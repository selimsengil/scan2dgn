# Start Qwen3.8-27B llama-server hidden, without cmd /c (so regex arguments with | survive). Log via --log-file.
# usage: powershell -File start_qwen2.ps1 -Args2 "<extra llama-server args>"
param(
    [string]$Model = "F:\Models\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-Q5_K_M.gguf",
    [string]$Mmproj = "F:\Models\Qwen3.8-27B-GGUF\mmproj-F16.gguf",
    [int]$Port = 8091,
    [string]$Log = "C:\Users\Selim\Documents\Proje\oda_kural\runs\qwen_server.log",
    [string]$Args2 = ""
)
$ll = "C:\Users\Selim\Documents\Proje\third_party\llama.cpp-b12110-cuda13.4\llama-server.exe"
Remove-Item $Log -ErrorAction SilentlyContinue
$env:CUDA_DEVICE_ORDER = "FASTEST_FIRST"   # inherited by the child (a Win32_ProcessStartup env list would replace the whole environment)
$p = Start-Process -FilePath $ll -ArgumentList "-m `"$Model`" --mmproj `"$Mmproj`" --port $Port --jinja --log-file `"$Log`" $Args2" -WindowStyle Hidden -PassThru
$p | Add-Member ProcessId $p.Id
"pid $($p.ProcessId)"
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep 5
    $t = Get-Content $Log -Raw -ErrorAction SilentlyContinue
    if ($t -match 'listening on') { "ready after $(($i + 1) * 5) s"; break }
    if ($t -match 'out of memory|exiting due to|failed to load model') { "PROBLEM"; break }
    if (-not (Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue)) { "process exited"; break }
}
Get-Content $Log -ErrorAction SilentlyContinue | Select-String -Pattern 'out of memory|model buffer size|KV buffer size|compute buffer size' | Select-Object -Last 12 | ForEach-Object { $_.Line.Trim() }
