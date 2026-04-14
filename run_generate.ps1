$env:PYTHONUTF8 = "1"
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\src\generate_dataset.py" @args
