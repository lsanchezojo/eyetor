# Eyetor Windows Service uninstaller
param([string]$Mode = "telegram")
powershell -ExecutionPolicy Bypass -File "$PSScriptRoot\install-service.ps1" -Mode $Mode -Action uninstall
