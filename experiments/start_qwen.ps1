# Start Qwen3.8-27B llama-server hidden (survives the session).
# usage: powershell -File start_qwen.ps1 -Model <gguf> -Split "16,6,5.5" -Devices "CUDA0,CUDA1,CUDA2" -Port 8091 -Log <file>
param(
    [string]$Model = "F:\Models\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-Q5_K_M.gguf",
    [string]$Mmproj = "F:\Models\Qwen3.8-27B-GGUF\mmproj-F16.gguf",
    [string]$Split = "16,6,5.5",
    [string]$Devices = "",
    [int]$Ctx = 32768,
    [int]$Port = 8091,
    [string]$Log = "C:\Users\Selim\Documents\Proje\oda_kural\runs\qwen_server.log",
    [string]$Extra = ""
)
$ll = "C:\Users\Selim\Documents\Proje\third_party\llama.cpp-b12110-cuda13.4\llama-server.exe"
$dev = if ($Devices) { "-dev $Devices" } else { "" }
$cmd = "cmd /c set `"CUDA_DEVICE_ORDER=FASTEST_FIRST`"&& `"$ll`" -m `"$Model`" --mmproj `"$Mmproj`" -ngl 99 $dev --tensor-split $Split -c $Ctx --port $Port --jinja $Extra > `"$Log`" 2>&1"
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow = [uint16]0}
$p = Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; ProcessStartupInformation = $si}
"pid $($p.ProcessId)"
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep 5
    $t = Get-Content $Log -Raw -ErrorAction SilentlyContinue
    if ($t -match 'listening on') { "ready after $(($i + 1) * 5) s"; break }
    if ($t -match 'out of memory|exiting due to|failed to load model') { "PROBLEM"; break }
}
Get-Content $Log | Select-String -Pattern 'CUDA\d.*(model buffer|KV buffer|compute buffer)|offloaded|using device' | Select-Object -Last 12 | ForEach-Object { $_.Line.Trim() }

