param(
    [Parameter(Mandatory = $true)][string]$ArtifactRoot,
    [Parameter(Mandatory = $true)][string]$ServiceAccount
)

$ErrorActionPreference = "Stop"
$paths = @(
    (Join-Path $ArtifactRoot "database"),
    (Join-Path $ArtifactRoot "evidence"),
    (Join-Path $ArtifactRoot "logs"),
    (Join-Path $ArtifactRoot "service")
)
$findings = @()
foreach ($path in $paths) {
    if (-not (Test-Path -LiteralPath $path)) {
        $findings += [pscustomobject]@{ path = $path; status = "missing" }
        continue
    }
    $acl = Get-Acl -LiteralPath $path
    $rules = $acl.Access | Where-Object { $_.IdentityReference.Value -ieq $ServiceAccount }
    if (-not $rules) {
        $findings += [pscustomobject]@{ path = $path; status = "no-explicit-service-account-rule" }
        continue
    }
    foreach ($rule in $rules) {
        $findings += [pscustomobject]@{
            path = $path
            identity = $rule.IdentityReference.Value
            rights = $rule.FileSystemRights.ToString()
            access = $rule.AccessControlType.ToString()
            inherited = $rule.IsInherited
            status = if ($rule.AccessControlType -eq "Deny") { "deny" } else { "allow" }
        }
    }
}
$findings | ConvertTo-Json -Depth 4
if ($findings | Where-Object { $_.status -eq "missing" -or $_.status -eq "no-explicit-service-account-rule" }) {
    exit 2
}
