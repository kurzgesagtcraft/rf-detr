# Get User Environment Variables
$userEnv = Get-ChildItem -Path HKCU:\Environment
$userEnv | ForEach-Object { "User: " + $_.Name + "=" + $_.GetValue($_.Name) } | Out-File -FilePath "env.txt" -Encoding utf8

# Get System Environment Variables
$systemEnv = Get-ChildItem -Path "HKLM:\System\CurrentControlSet\Control\Session Manager\Environment"
$systemEnv | ForEach-Object { "System: " + $_.Name + "=" + $_.GetValue($_.Name) } | Out-File -FilePath "env.txt" -Append -Encoding utf8