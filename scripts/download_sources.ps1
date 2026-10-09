$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$destinationRoot = Join-Path $projectRoot 'artifacts\sources\acquisition'
New-Item -ItemType Directory -Force -Path $destinationRoot | Out-Null
$files = @(
 @{Name='hackathonlicence.zip'; Url='http://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip'; Hash='A9F932FF4096A7DF797D1547987937F34D3995AC445B4748177114488D12B010'},
 @{Name='t_dict_municipal.rar'; Url='http://www.sberbank.com/common/files/t_dict_municipal.rar'; Hash='319ED22684B77716641BC15B61F7325ADC44E9FC47E9BE2973D88412D27D21F5'},
 @{Name='metadata_municipal_dict_sberindex_2.pdf'; Url='http://www.sberbank.ru/common/img/uploaded/files/pdf/sberindex/metadata_municipal_dict_sberindex_2.pdf'; Hash='05707427A64061552D1F23F74F578A99FDABEFF69F0C8A0FC99D924210FEE0B3'}
)
foreach ($item in $files) {
 $target = Join-Path $destinationRoot $item.Name
 if (-not (Test-Path -LiteralPath $target)) {
  Invoke-WebRequest -UseBasicParsing -Uri $item.Url -OutFile $target
 }
 $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash
 if ($actual -ne $item.Hash) { throw "Snapshot hash mismatch: $($item.Name). Do not extract; review source update." }
 Write-Output "Verified $($item.Name)"
}
# The two exact expected names are extracted from the verified RAR, not arbitrary members.
& tar -xf (Join-Path $destinationRoot 't_dict_municipal.rar') -C $destinationRoot t_dict_municipal_districts.xlsx t_dict_municipal_districts_poly.gpkg
if ($LASTEXITCODE -ne 0) { throw 'Windows tar RAR extraction failed. Use a trusted local archive utility for the two named files.' }
& python (Join-Path $PSScriptRoot 'unpack_sources.py')
if ($LASTEXITCODE -ne 0) { throw 'ZIP validation/extraction failed' }
