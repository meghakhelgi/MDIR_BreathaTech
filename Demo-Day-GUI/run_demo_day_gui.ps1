$ErrorActionPreference = "Stop"

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command py -ErrorAction SilentlyContinue
}

if (-not $python) {
    $candidatePaths = @(
        "C:\Users\Workstation\AppData\Local\Programs\Python\Python312\python.exe",
        "C:\Users\Workstation\AppData\Local\Programs\Python\Python311\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe"
    )

    foreach ($candidate in $candidatePaths) {
        if (Test-Path $candidate) {
            $python = [PSCustomObject]@{
                Name = Split-Path $candidate -Leaf
                Source = $candidate
            }
            break
        }
    }
}

if (-not $python) {
    throw "Python was not found on PATH and no known installation path was found. Install Python 3.11+ and try again."
}

if ($python.Name -eq "py.exe" -or $python.Name -eq "py") {
    & $python.Source -3 "$PSScriptRoot\app.py"
} else {
    & $python.Source "$PSScriptRoot\app.py"
}
