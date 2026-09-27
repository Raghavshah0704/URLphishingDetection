Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  PhiUSIIL Phishing Detector -- Browser Extension Backend API" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting REST API Server on http://localhost:5000 ..." -ForegroundColor Green
Write-Host "Keep this window running while using the Browser Extension!" -ForegroundColor Yellow
Write-Host ""
python api.py
