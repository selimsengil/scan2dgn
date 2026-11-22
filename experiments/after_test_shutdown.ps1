# Wait for the 7-room thinking test to finish, build the comparison page, then shut the PC down
# (the user asked for this before going to sleep). Gives up waiting after 45 min and shuts down anyway.
$log = "C:\Users\Selim\Documents\Proje\oda_kural\runs\qwenq4_teach_think7b.log"
$deadline = (Get-Date).AddMinutes(45)
while ((Get-Date) -lt $deadline) {
    if ((Get-Content $log -Raw -ErrorAction SilentlyContinue) -match "SUMMARY") { break }
    Start-Sleep 20
}
Set-Location "C:\Users\Selim\Documents\Proje\oda_kural"
& ..\.venv\Scripts\python.exe compare3.py "013_05,016_01,019_21,001_26,019_07,015_07,015_06" DUSUNME_7oda.html "qwenq4_fewshot132:few-shot" "qwenq4_teach132:öğretim seti" "qwenq4_teach_think7b:öğretim + düşünme 2500"
"page built $(Get-Date)" | Out-File runs\shutdown_note.txt -Encoding utf8
shutdown /s /t 120 /c "Qwen testi bitti, bilgisayar 2 dakika icinde kapanacak."
