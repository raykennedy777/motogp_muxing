<#
.SYNOPSIS
    Mux MotoGP 2026 multi-audio MKV files.

.PARAMETER Class
    Bike class: MotoGP, Moto2, or Moto3.

.PARAMETER Round
    Round number (1-22). Country is resolved automatically from the 2026 calendar.

.PARAMETER Session
    Session type: Race or Sprint.

.PARAMETER Channel
    Primary channel for the output filename (default: Polsat).

.PARAMETER Source
    Broadcast source for the output filename (default: HDTV).

.EXAMPLE
    .\2026_mux.ps1 -Class MotoGP -Round 1 -Session Sprint
    Outputs: MotoGP.2026.Round01.Thailand.Sprint.Polsat.HDTV.1080p.x265.Multi5.mkv
#>
param(
    [Parameter(Mandatory)][ValidateSet('MotoGP','Moto2','Moto3')][string]$Class,
    [Parameter(Mandatory)][ValidateRange(1,22)][int]$Round,
    [Parameter(Mandatory)][ValidateSet('Race','Sprint')][string]$Session,
    [string]$Channel = 'Polsat',
    [string]$Source  = 'HDTV'
)

# ── 2026 MotoGP Calendar (22 rounds) ─────────────────────────────────────────
$calendar = @{
     1 = 'Thailand';    2 = 'Brazil';       3 = 'USA';    4 = 'Qatar'
     5 = 'Spain.Jerez';  6 = 'France';       7 = 'Spain.Catalunya'; 8 = 'Italy'
     9 = 'Hungary';    10 = 'Czechia';        11 = 'Netherlands'; 12 = 'Germany'
    13 = 'Britain';    14 = 'Spain.Aragon';  15 = 'SanMarino';  16 = 'Austria'
    17 = 'Japan';      18 = 'Indonesia';    19 = 'Australia';   20 = 'Malaysia'
    21 = 'Portugal';   22 = 'Spain.Valencia'
}

$country  = $calendar[$Round]
$roundStr = $Round.ToString().PadLeft(2, '0')

# ── Locate source files ───────────────────────────────────────────────────────
$masterFile = Get-ChildItem -File -Filter '*master*' | Select-Object -First 1
$webFile    = Get-ChildItem -File |
              Where-Object { $_.Name -ilike '*web*' -and $_.Extension -in '.mkv','.mp4','.avi' } |
              Select-Object -First 1

if (-not $masterFile) { Write-Error "No file with 'master' in the name found."; exit 1 }
if (-not $webFile)    { Write-Error "No file with 'web' in the name found.";    exit 1 }

$tntFile  = 'en_tntsports.mka'
$daznFile = 'es_dazn.mka'

Write-Host "Master : $($masterFile.Name)"
Write-Host "Web    : $($webFile.Name)"

# ── Analyse master file (codec + resolution + Polsat audio track ID) ──────────
$masterJson  = & mkvmerge --identify --identification-format json $masterFile.FullName | ConvertFrom-Json
$videoTrack  = $masterJson.tracks | Where-Object type -eq 'video'  | Select-Object -First 1
$masterAudio = @($masterJson.tracks | Where-Object type -eq 'audio')

if (-not $videoTrack) { Write-Error "No video track found in '$($masterFile.Name)'."; exit 1 }

$codec = switch -Regex ($videoTrack.codec) {
    'HEVC' { 'x265'; break }
    'AVC'  { 'x264'; break }
    default { 'x264' }
}
$height     = [int]($videoTrack.properties.pixel_dimensions -split 'x')[1]
$resolution = "${height}p"

# ── Analyse web file (World Feed + Natural Sounds track IDs) ──────────────────
$webJson  = & mkvmerge --identify --identification-format json $webFile.FullName | ConvertFrom-Json
$webAudio = @($webJson.tracks | Where-Object type -eq 'audio')

# ── Determine which tracks are present ────────────────────────────────────────
$hasWF     = $webAudio.Count -ge 1
$hasNS     = $webAudio.Count -ge 2
$hasTNT    = Test-Path $tntFile
$hasDAZN   = Test-Path $daznFile
$hasPolsat = $masterAudio.Count -ge 1

$audioCount = ($hasWF, $hasTNT, $hasDAZN, $hasPolsat, $hasNS | Where-Object { $_ }).Count

$outputFile = "$Class.2026.Round$roundStr.$country.$Session.$Channel.$Source.$resolution.$codec.Multi$audioCount.mkv"

# ── Build mkvmerge argument list ──────────────────────────────────────────────
$mkv        = @('--output', $outputFile)
$trackOrder = @()
$idx        = 0

# Input 0 – master: VIDEO only (chapters from here)
$mkv += '--no-audio', '--no-subtitles', $masterFile.FullName
$trackOrder += "${idx}:$($videoTrack.id)"
$idx++

# World Feed – web file, audio track 1 (eng, default)
if ($hasWF) {
    $tid  = $webAudio[0].id
    $mkv += '--no-video', '--no-subtitles', '--no-chapters',
            '--audio-tracks',  "$tid",
            '--track-name',    "${tid}:World Feed",
            '--language',      "${tid}:eng",
            '--default-track', "${tid}:yes",
            $webFile.FullName
    $trackOrder += "${idx}:${tid}"
    $idx++
}

# TNT Sports – en_tntsports.mka, audio track 0 (eng)
if ($hasTNT) {
    $mkv += '--no-video', '--no-subtitles', '--no-chapters',
            '--audio-tracks',  '0',
            '--track-name',    '0:TNT Sports',
            '--language',      '0:eng',
            '--default-track', '0:no',
            $tntFile
    $trackOrder += "${idx}:0"
    $idx++
}

# DAZN – es_dazn.mka, audio track 0 (spa)
if ($hasDAZN) {
    $mkv += '--no-video', '--no-subtitles', '--no-chapters',
            '--audio-tracks',  '0',
            '--track-name',    '0:DAZN',
            '--language',      '0:spa',
            '--default-track', '0:no',
            $daznFile
    $trackOrder += "${idx}:0"
    $idx++
}

# Polsat Sport – master file, audio track 1 (pol)
if ($hasPolsat) {
    $tid  = $masterAudio[0].id
    $mkv += '--no-video', '--no-subtitles', '--no-chapters',
            '--audio-tracks',  "$tid",
            '--track-name',    "${tid}:Polsat Sport",
            '--language',      "${tid}:pol",
            '--default-track', "${tid}:no",
            $masterFile.FullName
    $trackOrder += "${idx}:${tid}"
    $idx++
}

# Natural Sounds – web file, audio track 2 (und/no language)
if ($hasNS) {
    $tid  = $webAudio[1].id
    $mkv += '--no-video', '--no-subtitles', '--no-chapters',
            '--audio-tracks',  "$tid",
            '--track-name',    "${tid}:Natural Sounds",
            '--language',      "${tid}:und",
            '--default-track', "${tid}:no",
            $webFile.FullName
    $trackOrder += "${idx}:${tid}"
    $idx++
}

$mkv += '--track-order', ($trackOrder -join ',')

# ── Summary + run ─────────────────────────────────────────────────────────────
Write-Host "`nOutput     : $outputFile"
Write-Host "Codec      : $codec  |  Resolution: $resolution  |  Audio tracks: $audioCount"
Write-Host "Track order: $($trackOrder -join ', ')`n"

& mkvmerge @mkv
