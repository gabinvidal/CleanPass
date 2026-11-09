# CleanPass - Windows build script
# Builds CleanPass.dll for every Nuke 16 / 17 found in "C:\Program Files"
# (or for the versions given with -Nuke), and bundles the OIDN runtime.
#
#   powershell -ExecutionPolicy Bypass -File build.ps1
#   powershell -ExecutionPolicy Bypass -File build.ps1 -Nuke "C:\Program Files\Nuke17.0v3"
#
# Needs Visual Studio 2022 (or its Build Tools) with the "Desktop development
# with C++" workload. CMake is taken from Visual Studio if it is not on PATH.
# The OIDN 2.5.1 SDK is downloaded to deps\ the first time.
#
# Private OIDN runtime
# --------------------
# Nuke ships an older C++ runtime (msvcp140 14.36) and its own TBB. Recent
# OIDN builds need a newer msvcp140 (>= 14.40) and a newer TBB, and other
# plugins may load their own OIDN. So the bundled runtime gets private file
# names (OIDN_CleanPass25*.dll, tbbCP.dll, MSVCP_CP.dll): the import names
# inside the DLLs are renamed to match (same length, nothing else changes).
# MSVCP_CP.dll is your own Visual Studio's msvcp140.dll, copied at build time.

param(
    [string[]]$Nuke = @(),
    [int[]]$Majors = @(16, 17),
    [string]$OidnRoot = $env:OIDN_ROOT
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Fail($msg) { Write-Host "`nBUILD FAILED: $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- Visual Studio
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsPath = $null
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -products * -version '[17.0,18.0)' `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if (-not $vsPath) {
    Fail ("Visual Studio 2022 with the C++ tools was not found.`n" +
          "Install 'Build Tools for Visual Studio 2022' (free) and tick 'Desktop development with C++':`n" +
          "https://visualstudio.microsoft.com/visual-cpp-build-tools/")
}
Write-Host "Visual Studio: $vsPath"

$cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmake) {
    $cand = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    if (Test-Path $cand) { $cmake = $cand }
}
if (-not $cmake) { Fail "CMake not found (install it, or add the 'C++ CMake tools' component to Visual Studio)." }
Write-Host "CMake: $cmake"

# ----------------------------------------------------------------- OIDN 2.5.1
$OidnVersion = '2.5.1'
if (-not $OidnRoot) { $OidnRoot = Join-Path $root "deps\oidn-$OidnVersion.x64.windows" }
if (-not (Test-Path (Join-Path $OidnRoot 'include\OpenImageDenoise\oidn.hpp'))) {
    $deps = Join-Path $root 'deps'
    New-Item -ItemType Directory -Force -Path $deps | Out-Null
    $zip = Join-Path $deps "oidn-$OidnVersion.x64.windows.zip"
    $url = "https://github.com/RenderKit/oidn/releases/download/v$OidnVersion/oidn-$OidnVersion.x64.windows.zip"
    Write-Host "Downloading OIDN $OidnVersion SDK..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $deps -Force
    Remove-Item $zip
    $OidnRoot = Join-Path $deps "oidn-$OidnVersion.x64.windows"
}
Write-Host "OIDN: $OidnRoot"

# bundle the OIDN runtime next to the plugin (shared by every Nuke version), under private names
$latin = [Text.Encoding]::GetEncoding(28591)   # byte-exact string <-> bytes
function Write-Patched([string]$Src, [string]$Dst, [object[]]$Pairs) {
    $text = $latin.GetString([IO.File]::ReadAllBytes($Src))
    foreach ($p in $Pairs) {
        $from = $p[0]; $to = $p[1]
        if ($from.Length -ne $to.Length) { Fail "internal: '$from' / '$to' lengths differ" }
        if ($text.IndexOf($from, [StringComparison]::Ordinal) -lt 0) {
            if ($p.Count -ge 3) { continue }   # optional
            Fail "'$($from.TrimEnd([char]0))' not found in $(Split-Path $Src -Leaf) (unexpected OIDN build)"
        }
        $text = $text.Replace($from, $to)
    }
    $bytes = $latin.GetBytes($text)
    if ((Test-Path $Dst) -and ((Get-FileHash -InputStream ([IO.MemoryStream]::new($bytes))).Hash -eq (Get-FileHash $Dst).Hash)) { return }
    try { [IO.File]::WriteAllBytes($Dst, $bytes) }
    catch { Write-Host "Could not update $(Split-Path $Dst -Leaf) (in use by Nuke?) - keeping the current copy." -ForegroundColor Yellow }
}
function Find-Msvcp {
    $cands = @()
    $redist = Join-Path $vsPath 'VC\Redist\MSVC'
    if (Test-Path $redist) {
        $cands += Get-ChildItem -Recurse -Path $redist -Filter 'msvcp140.dll' -ErrorAction SilentlyContinue |
                  Where-Object { $_.FullName -match '\\x64\\Microsoft\.VC\d+\.CRT\\' }
    }
    $sys = Join-Path $env:SystemRoot 'System32\msvcp140.dll'
    if (Test-Path $sys) { $cands += Get-Item $sys }
    $best = $null; $bestV = [version]'0.0'
    foreach ($c in $cands) {
        $vi = $c.VersionInfo
        $v = [version]("{0}.{1}.{2}.{3}" -f $vi.FileMajorPart, $vi.FileMinorPart, $vi.FileBuildPart, $vi.FilePrivatePart)
        if ($v -gt $bestV) { $best = $c.FullName; $bestV = $v }
    }
    if ($bestV -lt [version]'14.40') { return $null }
    return $best
}
$oidnBin = Join-Path $root 'oidn\bin'
New-Item -ItemType Directory -Force -Path $oidnBin | Out-Null
$P = 'OIDN_CleanPass25'                       # same length as 'OpenImageDenoise'
$core = @("OpenImageDenoise_core.dll", "$($P)_core.dll")
$crt = @('MSVCP140.dll', 'MSVCP_CP.dll', 'optional')
$tbb = @('tbb12.dll', 'tbbCP.dll')
$ob = Join-Path $OidnRoot 'bin'
Write-Patched (Join-Path $ob 'OpenImageDenoise.dll') (Join-Path $oidnBin "$P.dll") @($core, $crt)
Write-Patched (Join-Path $ob 'OpenImageDenoise_core.dll') (Join-Path $oidnBin "$($P)_core.dll") @(@("OpenImageDenoise_$([char]0)", "$($P)_$([char]0)"), @($core[0], $core[1], 'optional'), $crt)
Write-Patched (Join-Path $ob 'OpenImageDenoise_device_cpu.dll') (Join-Path $oidnBin "$($P)_device_cpu.dll") @($core, $tbb, $crt)
Write-Patched (Join-Path $ob 'OpenImageDenoise_device_cuda.dll') (Join-Path $oidnBin "$($P)_device_cuda.dll") @($core, $crt)
Write-Patched (Join-Path $ob 'tbb12.dll') (Join-Path $oidnBin 'tbbCP.dll') @(@('tbb12.dll', 'tbbCP.dll', 'optional'), $crt)
$msvcp = Find-Msvcp
if (-not $msvcp) {
    Fail ("No msvcp140.dll 14.40 or newer found (in Visual Studio's redist folder or System32).`n" +
          "Update Visual Studio 2022 / its Build Tools, or install the latest Visual C++ Redistributable.")
}
$dstCrt = Join-Path $oidnBin 'MSVCP_CP.dll'
if (-not ((Test-Path $dstCrt) -and ((Get-FileHash $msvcp).Hash -eq (Get-FileHash $dstCrt).Hash))) {
    try { Copy-Item $msvcp $dstCrt -Force } catch { Write-Host "Could not update MSVCP_CP.dll (in use by Nuke?)." -ForegroundColor Yellow }
}
Write-Host "OIDN runtime: $oidnBin (C++ runtime from $msvcp)"
# licences of the redistributed runtime (Apache 2.0 + third-party notices)
foreach ($lic in Get-ChildItem -Path (Join-Path $OidnRoot 'doc') -Filter '*.txt' -ErrorAction SilentlyContinue) {
    Copy-Item $lic.FullName (Join-Path $root 'oidn') -Force
}

# --------------------------------------------------------------- Nuke versions
function Get-NukeVersionKey($dir) {
    if ((Split-Path $dir -Leaf) -match '^Nuke(\d+)\.(\d+)v(\d+)') {
        return [int]$matches[1] * 1000000 + [int]$matches[2] * 1000 + [int]$matches[3]
    }
    return 0
}
if ($Nuke.Count -eq 0) {
    foreach ($m in $Majors) {
        $found = Get-ChildItem -Directory -Path $env:ProgramFiles -Filter "Nuke$m.*" -ErrorAction SilentlyContinue |
                 Where-Object { Test-Path (Join-Path $_.FullName 'cmake\NukeConfig.cmake') } |
                 Sort-Object { Get-NukeVersionKey $_.FullName } | Select-Object -Last 1
        if ($found) { $Nuke += $found.FullName } else { Write-Host "Nuke $m not found, skipped." -ForegroundColor Yellow }
    }
}
if ($Nuke.Count -eq 0) { Fail "No Nuke install found. Pass one with -Nuke `"C:\Program Files\Nuke17.0v3`"." }

$built = @()
foreach ($n in $Nuke) {
    if (-not ((Split-Path $n -Leaf) -match '^Nuke(\d+)\.')) { Fail "Cannot read the Nuke version from '$n'." }
    $major = $matches[1]
    $bdir = Join-Path $root "build\nuke$major"
    Write-Host "`n=== Nuke $major  ($n) ===" -ForegroundColor Cyan
    & $cmake -S $root -B $bdir -G 'Visual Studio 17 2022' -A x64 "-DNUKE_ROOT=$($n -replace '\\','/')" "-DOIDN_ROOT=$($OidnRoot -replace '\\','/')"
    if ($LASTEXITCODE -ne 0) { Fail "CMake configure failed for Nuke $major." }
    & $cmake --build $bdir --config Release
    if ($LASTEXITCODE -ne 0) { Fail "Compilation failed for Nuke $major." }
    $out = Get-ChildItem -Recurse -Path $bdir -Filter 'CleanPass.dll' | Select-Object -First 1
    if (-not $out) { Fail "CleanPass.dll not produced for Nuke $major." }
    $dst = Join-Path $root "bin\$major"
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    # link to the private OIDN runtime (see the top of this file)
    Write-Patched $out.FullName (Join-Path $dst 'CleanPass.dll') @(, @('OpenImageDenoise.dll', "$P.dll"))
    $pdb = [IO.Path]::ChangeExtension($out.FullName, '.pdb')
    if (Test-Path $pdb) { try { Copy-Item $pdb $dst -Force } catch { Write-Host "Could not update the .pdb (in use)." -ForegroundColor Yellow } }
    $built += "bin\$major\CleanPass.dll"
}

Write-Host "`nOK - built:" -ForegroundColor Green
$built | ForEach-Object { Write-Host "  $_" }
Write-Host "Restart Nuke and Tab-search 'CleanPass' (or Filter > CleanPass)."
