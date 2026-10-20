# Script de compilação da Monografia em LaTeX (ABNT/abnTeX2)
# Não necessita de Perl ou latexmk

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "COMPILANDO MONOGRAFIA DE TCC (abnTeX2 + BibTeX)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "`n[1/4] Primeira passagem do pdfLaTeX..." -ForegroundColor Yellow
pdflatex -interaction=nonstopmode main.tex | Out-Null

Write-Host "[2/4] Processando referências bibliográficas com BibTeX..." -ForegroundColor Yellow
bibtex main | Out-Null

Write-Host "[3/4] Segunda passagem do pdfLaTeX (resolvendo citações)..." -ForegroundColor Yellow
pdflatex -interaction=nonstopmode main.tex | Out-Null

Write-Host "[4/4] Terceira passagem do pdfLaTeX (ajustando sumário e referências cruzadas)..." -ForegroundColor Yellow
pdflatex -interaction=nonstopmode main.tex | Out-Null

if (Test-Path "main.pdf") {
    $tamanhoKB = [math]::Round((Get-Item "main.pdf").Length / 1KB, 1)
    Write-Host "`n[SUCESSO] Monografia gerada com sucesso: main.pdf ($tamanhoKB KB)!" -ForegroundColor Green
} else {
    Write-Host "`n[ERRO] Não foi possível gerar main.pdf. Verifique o arquivo main.log." -ForegroundColor Red
}
