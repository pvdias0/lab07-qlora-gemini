$env:PYTHONUTF8 = "1"
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\src\train_qlora.py" @args
