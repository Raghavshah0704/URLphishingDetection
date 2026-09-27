Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "     PhiUSIIL Phishing URL Detector -- Web Dashboard" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Launching Streamlit web application..." -ForegroundColor Green
Write-Host "Opening in browser at http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
python -m streamlit run app.py
