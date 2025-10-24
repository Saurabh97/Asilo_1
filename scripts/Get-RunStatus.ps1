param(
  [string]$IndexPath,                 # optional: specific outputs\run_index_*.json
  [string]$ReportPath                 # optional: write summary to this file
)

# 1) Pick latest index if not provided
if (-not $IndexPath) {
  $idxItem = Get-ChildItem outputs\run_index_*.json -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $idxItem) { Write-Error "No outputs\run_index_*.json found."; exit 1 }
  $IndexPath = $idxItem.FullName
}

$runs = Get-Content $IndexPath | ConvertFrom-Json

function Test-RunComplete {
  param([string]$ExpName)

  # A) done marker
  $marker = Join-Path "outputs\completed" "$ExpName.done.json"
  if (Test-Path $marker) { return $true }

  # B) .out log with "rounds complete"
  $exact = Join-Path "outputs\run_stdout" "$ExpName.out"
  $outPath = $null
  if (Test-Path $exact) {
    $outPath = $exact
  } else {
    # wildcard to tolerate small naming differences
    $cand = Get-ChildItem "outputs\run_stdout\*$ExpName*.out" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cand) { $outPath = $cand.FullName }
  }
  if ($outPath -and (Select-String -Path $outPath -Pattern "rounds complete" -Quiet)) {
    return $true
  }
  return $false
}

$completed = @()
$pending   = @()

foreach ($r in $runs) {
  if (Test-RunComplete $r.exp_name) { $completed += $r } else { $pending += $r }
}

# Build summary text
$lines = @()
$lines += "Run index : $([System.IO.Path]::GetFileName($IndexPath))"
$lines += "Total     : $($runs.Count)"
$lines += "Completed : $($completed.Count)"
$lines += "Pending   : $($pending.Count)"
$lines += ""
$lines += "=== Completed by method ==="
$lines += ($completed | Group-Object method | Sort-Object Name | ForEach-Object { "{0,-10} : {1}" -f $_.Name, $_.Count })
if (-not $completed) { $lines += "(none)" }
$lines += ""
$lines += "=== Pending by method ==="
$lines += ($pending | Group-Object method | Sort-Object Name | ForEach-Object { "{0,-10} : {1}" -f $_.Name, $_.Count })
if (-not $pending) { $lines += "(none)" }
$lines += ""
$lines += "=== Pending examples (up to 20) ==="
$lines += ($pending | Select-Object -First 20 | ForEach-Object { "{0}  {1,-8} seed={2}" -f $_.exp_name, $_.method, $_.seed })
if (-not $pending) { $lines += "(none)" }

# Emit and optionally write report
$report = $lines -join [Environment]::NewLine
$report

if ($ReportPath) {
  $dir = Split-Path $ReportPath -Parent
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  $report | Out-File -FilePath $ReportPath -Encoding UTF8
}
