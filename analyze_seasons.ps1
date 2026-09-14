# 分析同一资源不同季的SID前缀
param(
    [string]$InputFile = "results_three_way\exp_task3_l1_sinkhorn\indices\indices.jsonl",
    [string]$OutputFile = "同季prefix分析报告.md"
)

Write-Host "开始分析多季资源的SID前缀..." -ForegroundColor Green

$seasonData = @{}
$lineCount = 0

Get-Content $InputFile | ForEach-Object {
    $lineCount++
    if ($lineCount % 10000 -eq 0) {
        Write-Host "已处理 $lineCount 行..." -ForegroundColor Yellow
    }
    
    $d = $_ | ConvertFrom-Json
    $season = $d.season
    
    if ($season) {
        $name = $d.meta_main_name
        $tokens = $d.tokens
        
        if (-not $seasonData.ContainsKey($name)) {
            $seasonData[$name] = @()
        }
        
        $seasonData[$name] += @{
            season = $season
            tokens = $tokens
            L1 = $tokens[0]
            L2 = $tokens[1]
            L3 = $tokens[2]
            L1L2 = "$($tokens[0]) $($tokens[1])"
            L1L2L3 = "$($tokens[0]) $($tokens[1]) $($tokens[2])"
            full = ($tokens -join ' ')
        }
    }
}

Write-Host "数据加载完成，共 $lineCount 行" -ForegroundColor Green

# 筛选多季资源
$multiSeason = $seasonData.GetEnumerator() | Where-Object { $_.Value.Count -gt 1 } | Sort-Object { $_.Value.Count } -Descending

Write-Host "找到 $($multiSeason.Count) 个有多季的资源" -ForegroundColor Green

# 分析prefix相同情况
$stats = @{
    total = $multiSeason.Count
    L1_same = 0
    L1L2_same = 0
    L1L2L3_same = 0
    all_different = 0
}

$examples = @{
    L1_same = @()
    L1L2_same = @()
    L1L2L3_same = @()
    all_different = @()
}

foreach ($item in $multiSeason) {
    $name = $item.Key
    $seasons = $item.Value | Sort-Object { [int]$_.season }
    
    # 检查L1是否相同
    $L1s = $seasons | ForEach-Object { $_.L1 } | Select-Object -Unique
    $L1L2s = $seasons | ForEach-Object { $_.L1L2 } | Select-Object -Unique
    $L1L2L3s = $seasons | ForEach-Object { $_.L1L2L3 } | Select-Object -Unique
    
    if ($L1s.Count -eq 1) {
        $stats.L1_same++
        if ($examples.L1_same.Count -lt 5) {
            $examples.L1_same += @{ name=$name; seasons=$seasons }
        }
        
        if ($L1L2s.Count -eq 1) {
            $stats.L1L2_same++
            if ($examples.L1L2_same.Count -lt 5) {
                $examples.L1L2_same += @{ name=$name; seasons=$seasons }
            }
            
            if ($L1L2L3s.Count -eq 1) {
                $stats.L1L2L3_same++
                if ($examples.L1L2L3_same.Count -lt 5) {
                    $examples.L1L2L3_same += @{ name=$name; seasons=$seasons }
                }
            }
        }
    } else {
        $stats.all_different++
        if ($examples.all_different.Count -lt 5) {
            $examples.all_different += @{ name=$name; seasons=$seasons }
        }
    }
}

# 生成报告
$report = @"
# 同一资源不同季的SID前缀分析报告

---

## 📊 统计总览

- **分析数据**: $InputFile
- **总数据量**: $lineCount 条
- **有多季的资源**: $($stats.total) 个
- **总季数**: $($multiSeason | ForEach-Object { $_.Value.Count } | Measure-Object -Sum | Select-Object -ExpandProperty Sum) 季

---

## 🎯 核心发现

### Prefix相同性统计

| Prefix层级 | 相同数量 | 比例 | 说明 |
|---|---:|---:|---|
| **L1相同** | $($stats.L1_same) | $('{0:P2}' -f ($stats.L1_same/$stats.total)) | 第1层token相同 |
| **L1L2相同** | $($stats.L1L2_same) | $('{0:P2}' -f ($stats.L1L2_same/$stats.total)) | 前2层token相同 |
| **L1L2L3相同** | $($stats.L1L2L3_same) | $('{0:P2}' -f ($stats.L1L2L3_same/$stats.total)) | 前3层token相同 |
| **完全不同** | $($stats.all_different) | $('{0:P2}' -f ($stats.all_different/$stats.total)) | L1就不同 |

---

## 📝 典型案例

### 案例1: L1相同的资源 (前5个)

"@

foreach ($ex in $examples.L1_same) {
    $report += "`n**资源名**: $($ex.name) ($($ex.seasons.Count)季)`n`n"
    foreach ($s in $ex.seasons) {
        $report += "- 第$($s.season)季: ``$($s.full)```n"
    }
    $report += "`n"
}

$report += @"

### 案例2: L1L2相同的资源 (前5个)

"@

foreach ($ex in $examples.L1L2_same) {
    $report += "`n**资源名**: $($ex.name) ($($ex.seasons.Count)季)`n`n"
    foreach ($s in $ex.seasons) {
        $report += "- 第$($s.season)季: ``$($s.full)```n"
    }
    $report += "`n"
}

$report += @"

### 案例3: L1L2L3相同的资源 (前5个)

"@

foreach ($ex in $examples.L1L2L3_same) {
    $report += "`n**资源名**: $($ex.name) ($($ex.seasons.Count)季)`n`n"
    foreach ($s in $ex.seasons) {
        $report += "- 第$($s.season)季: ``$($s.full)```n"
    }
    $report += "`n"
}

$report += @"

### 案例4: 完全不同的资源 (前5个)

"@

foreach ($ex in $examples.all_different) {
    $report += "`n**资源名**: $($ex.name) ($($ex.seasons.Count)季)`n`n"
    foreach ($s in $ex.seasons) {
        $report += "- 第$($s.season)季: ``$($s.full)```n"
    }
    $report += "`n"
}

$report += @"

---

## 🔍 深入分析

### 季数分布 (Top 20)

| 资源名 | 季数 | L1相同? |
|---|---:|:---:|
"@

$top20 = $multiSeason | Select-Object -First 20
foreach ($item in $top20) {
    $name = $item.Key
    $count = $item.Value.Count
    $L1s = $item.Value | ForEach-Object { $_.L1 } | Select-Object -Unique
    $L1Same = if ($L1s.Count -eq 1) { "✅" } else { "❌" }
    $report += "`n| $name | $count | $L1Same |"
}

$report += @"


---

## 💡 结论

### 主要发现

1. **L1相同率**: $('{0:P2}' -f ($stats.L1_same/$stats.total)) 的多季资源L1 token相同
   - 说明模型能较好地将同系列资源分配到相同的顶层类别

2. **L1L2相同率**: $('{0:P2}' -f ($stats.L1L2_same/$stats.total)) 的资源前2层相同
   - 显示了模型对系列资源的层级聚类能力

3. **L1L2L3相同率**: $('{0:P2}' -f ($stats.L1L2L3_same/$stats.total)) 的资源前3层相同
   - 前3层相同表明高度相似的语义表示

4. **完全不同**: $('{0:P2}' -f ($stats.all_different/$stats.total)) 的资源连L1都不同
   - 可能是跨类型的系列（如纪录片vs电影）

### 实际意义

- **语义一致性**: L1相同说明模型能识别系列资源的共同主题
- **层级区分**: 通过L2、L3、L4、L5逐层细化，区分不同季
- **检索友好**: 同系列资源前缀相近，有利于前缀搜索和推荐

---

**生成时间**: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  
**数据来源**: $InputFile
"@

$report | Out-File -FilePath $OutputFile -Encoding utf8
Write-Host "`n报告已保存到: $OutputFile" -ForegroundColor Green
Write-Host "分析完成！" -ForegroundColor Green

