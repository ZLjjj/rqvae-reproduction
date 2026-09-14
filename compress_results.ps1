# compress_results.ps1
# 用途: 一键压缩训练结果，准备上传到GitHub Release

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputFile = "results_rqvae_l1_sinkhorn_$timestamp.zip"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  GenSearchRec 训练结果压缩工具" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] 检查源目录..." -ForegroundColor Yellow
if (-not (Test-Path "results_rqvae_l1_sinkhorn")) {
    Write-Host "❌ 错误: 找不到 results_rqvae_l1_sinkhorn 目录" -ForegroundColor Red
    exit 1
}

$totalSize = (Get-ChildItem results_rqvae_l1_sinkhorn -Recurse -File | Measure-Object -Property Length -Sum).Sum
$totalSizeGB = [math]::Round($totalSize / 1GB, 2)
Write-Host "   源目录大小: $totalSizeGB GB" -ForegroundColor Green
Write-Host ""

Write-Host "[2/3] 开始压缩 (这可能需要几分钟)..." -ForegroundColor Yellow
$startTime = Get-Date

try {
    Compress-Archive -Path results_rqvae_l1_sinkhorn `
        -DestinationPath $outputFile `
        -CompressionLevel Optimal `
        -Force
    
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds
    
    Write-Host "   ✅ 压缩完成! 用时: $([math]::Round($duration, 2)) 秒" -ForegroundColor Green
} catch {
    Write-Host "   ❌ 压缩失败: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

Write-Host "[3/3] 检查压缩包..." -ForegroundColor Yellow
if (Test-Path $outputFile) {
    $compressedSize = (Get-Item $outputFile).Length
    $compressedSizeGB = [math]::Round($compressedSize / 1GB, 2)
    $ratio = [math]::Round(($compressedSize / $totalSize) * 100, 2)
    
    Write-Host "   文件名: $outputFile" -ForegroundColor Green
    Write-Host "   压缩后: $compressedSizeGB GB" -ForegroundColor Green
    Write-Host "   压缩率: $ratio%" -ForegroundColor Green
    Write-Host ""
    
    if ($compressedSizeGB -gt 2) {
        Write-Host "⚠️  警告: 文件超过2GB，无法作为单个Release上传" -ForegroundColor Yellow
        Write-Host "   建议: 分别压缩 task3 和 task3_correct" -ForegroundColor Yellow
    } else {
        Write-Host "✅ 文件大小符合GitHub Release限制 (< 2GB)" -ForegroundColor Green
    }
} else {
    Write-Host "   ❌ 找不到压缩包" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  下一步操作:" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "1. 创建GitHub Release:"
Write-Host "   gh release create v1.0.0 $outputFile --title 'Training Results' --notes '完整训练结果'" -ForegroundColor White
Write-Host ""
Write-Host "2. 或手动上传到:"
Write-Host "   https://github.com/你的用户名/仓库名/releases/new" -ForegroundColor White
Write-Host ""
