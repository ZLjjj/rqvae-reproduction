# 任务3+L1 Sinkhorn 完整5层分析
$dataFile = "results_three_way\exp_task3_l1_sinkhorn\indices\indices.jsonl"
$outputFile = "完整5层SID分析_任务3_L1L4_Sinkhorn.md"

Write-Host "开始分析任务3+L1 Sinkhorn的完整5层SID..."

# 数据结构
$seasonData = @{}
$totalLines = 0
$l5Distribution = @{}

# 读取数据
Get-Content $dataFile -Encoding utf8 | ForEach-Object {
    $totalLines++
    if ($totalLines % 10000 -eq 0) {
        Write-Host "已处理 $totalLines 行..."
    }
    
    $d = $_ | ConvertFrom-Json
    $season = $d.season
    
    if ($season) {
        $name = $d.meta_main_name
        if (-not $name) { return }
        
        # 提取5层tokens
        $tokens = $d.tokens
        if ($tokens.Count -ge 5) {
            $l1 = $tokens[0]
            $l2 = $tokens[1]
            $l3 = $tokens[2]
            $l4 = $tokens[3]
            $l5 = $tokens[4]
            
            # 统计L5分布
            if (-not $l5Distribution.ContainsKey($l5)) {
                $l5Distribution[$l5] = 0
            }
            $l5Distribution[$l5]++
            
            if (-not $seasonData.ContainsKey($name)) {
                $seasonData[$name] = @()
            }
            
            $seasonData[$name] += @{
                season = $season
                l1 = $l1
                l2 = $l2
                l3 = $l3
                l4 = $l4
                l5 = $l5
                sid = $tokens -join ' '
            }
        }
    }
}

Write-Host "数据加载完成，共 $totalLines 行"
Write-Host "找到 $($seasonData.Count) 个有多季的资源"

# 分析前缀相同性
$l1Same = 0
$l1l2Same = 0
$l1l2l3Same = 0
$l1l2l3l4Same = 0
$l1l2l3l4l5Same = 0
$allDifferent = 0

$l1SameResources = @()
$l1l2SameResources = @()
$l1l2l3SameResources = @()
$l1l2l3l4SameResources = @()
$l1l2l3l4l5SameResources = @()
$allDifferentResources = @()

foreach ($name in $seasonData.Keys) {
    $seasons = $seasonData[$name]
    if ($seasons.Count -le 1) { continue }
    
    $l1s = $seasons | ForEach-Object { $_.l1 } | Select-Object -Unique
    $l2s = $seasons | ForEach-Object { "$($_.l1) $($_.l2)" } | Select-Object -Unique
    $l3s = $seasons | ForEach-Object { "$($_.l1) $($_.l2) $($_.l3)" } | Select-Object -Unique
    $l4s = $seasons | ForEach-Object { "$($_.l1) $($_.l2) $($_.l3) $($_.l4)" } | Select-Object -Unique
    $l5s = $seasons | ForEach-Object { "$($_.l1) $($_.l2) $($_.l3) $($_.l4) $($_.l5)" } | Select-Object -Unique
    
    if ($l5s.Count -eq 1) {
        $l1l2l3l4l5Same++
        $l1l2l3l4l5SameResources += @{name=$name; seasons=$seasons}
    }
    elseif ($l4s.Count -eq 1) {
        $l1l2l3l4Same++
        $l1l2l3l4SameResources += @{name=$name; seasons=$seasons}
    }
    elseif ($l3s.Count -eq 1) {
        $l1l2l3Same++
        $l1l2l3SameResources += @{name=$name; seasons=$seasons}
    }
    elseif ($l2s.Count -eq 1) {
        $l1l2Same++
        $l1l2SameResources += @{name=$name; seasons=$seasons}
    }
    elseif ($l1s.Count -eq 1) {
        $l1Same++
        $l1SameResources += @{name=$name; seasons=$seasons}
    }
    else {
        $allDifferent++
        $allDifferentResources += @{name=$name; seasons=$seasons}
    }
}

# 生成报告
$report = @"
# 任务3+L1 Sinkhorn 完整5层SID分析报告

---

## 📊 统计总览

- **分析数据**: $dataFile
- **总数据量**: $totalLines 条
- **有多季的资源**: $($seasonData.Count) 个
- **总季数**: $(($seasonData.Values | ForEach-Object { $_.Count }) | Measure-Object -Sum).Sum 季

---

## 🎯 核心发现：完整5层前缀相同性

| Prefix层级 | 相同数量 | 比例 | 说明 |
|---|---:|---:|---|
| **L1相同** | $l1Same | $([math]::Round($l1Same/$seasonData.Count*100,2))% | 仅第1层token相同 |
| **L1L2相同** | $l1l2Same | $([math]::Round($l1l2Same/$seasonData.Count*100,2))% | 前2层token相同 |
| **L1L2L3相同** | $l1l2l3Same | $([math]::Round($l1l2l3Same/$seasonData.Count*100,2))% | 前3层token相同 |
| **L1L2L3L4相同** | $l1l2l3l4Same | $([math]::Round($l1l2l3l4Same/$seasonData.Count*100,2))% | 前4层token相同 |
| **L1L2L3L4L5完全相同** | $l1l2l3l4l5Same | $([math]::Round($l1l2l3l4l5Same/$seasonData.Count*100,2))% | 5层完全相同！ |
| **完全不同** | $allDifferent | $([math]::Round($allDifferent/$seasonData.Count*100,2))% | L1就不同 |

---

## 📝 典型案例

"@

# 案例1: 5层完全相同
$report += "`n### 案例1: 5层完全相同的资源 (前10个)`n`n"
$count = 0
foreach ($item in $l1l2l3l4l5SameResources) {
    if ($count -ge 10) { break }
    $count++
    $report += "**资源名**: $($item.name) ($($item.seasons.Count)季)`n`n"
    $report += "- **完全相同的SID**: ``$($item.seasons[0].sid)```n"
    $report += "- **所有季**: "
    $report += ($item.seasons | ForEach-Object { "第$($_.season)季" }) -join ", "
    $report += "`n`n"
}

# 案例2: L1L2L3L4相同
$report += "`n### 案例2: L1L2L3L4相同的资源 (前10个)`n`n"
$count = 0
foreach ($item in $l1l2l3l4SameResources) {
    if ($count -ge 10) { break }
    $count++
    $report += "**资源名**: $($item.name) ($($item.seasons.Count)季)`n`n"
    foreach ($s in $item.seasons | Select-Object -First 5) {
        $report += "- 第$($s.season)季: ``$($s.sid)```n"
    }
    $report += "`n"
}

# 案例3: L1L2L3相同
$report += "`n### 案例3: L1L2L3相同的资源 (前10个)`n`n"
$count = 0
foreach ($item in $l1l2l3SameResources) {
    if ($count -ge 10) { break }
    $count++
    $report += "**资源名**: $($item.name) ($($item.seasons.Count)季)`n`n"
    foreach ($s in $item.seasons | Select-Object -First 5) {
        $report += "- 第$($s.season)季: ``$($s.sid)```n"
    }
    $report += "`n"
}

# L5分布统计
$report += @"

---

## 🔢 L5层Token分布统计

L5是最后一层，通常编码最细粒度的特征。

### L5使用频率 Top 20

| L5 Token | 出现次数 | 占比 |
|---|---:|---:|
"@

$l5Sorted = $l5Distribution.GetEnumerator() | Sort-Object -Property Value -Descending | Select-Object -First 20
foreach ($item in $l5Sorted) {
    $pct = [math]::Round($item.Value / $totalLines * 100, 2)
    $report += "| ``$($item.Key)`` | $($item.Value) | $pct% |`n"
}

# 季数分布
$report += @"

---

## 🔍 多季资源季数分布 (Top 30)

| 资源名 | 季数 | 前缀相同性 |
|---|---:|:---:|
"@

$seasonCounts = @()
foreach ($name in $seasonData.Keys) {
    $seasons = $seasonData[$name]
    $l1s = $seasons | ForEach-Object { $_.l1 } | Select-Object -Unique
    
    $status = if ($l1s.Count -eq 1) { "✅ L1+" } else { "❌" }
    
    $seasonCounts += @{
        name = $name
        count = $seasons.Count
        status = $status
    }
}

$seasonCounts = $seasonCounts | Sort-Object -Property count -Descending | Select-Object -First 30
foreach ($item in $seasonCounts) {
    $report += "| $($item.name) | $($item.count) | $($item.status) |`n"
}

# 结论
$report += @"

---

## 💡 结论与洞察

### 主要发现

1. **5层完全相同率**: $([math]::Round($l1l2l3l4l5Same/$seasonData.Count*100,2))% 的多季资源5层SID完全相同
   - 这些资源的不同季被分配了**完全相同的语义标识**
   - 意味着模型无法区分这些资源的不同季
   - 可能原因：元数据相似度极高、内容高度重复

2. **L1L2L3L4相同率**: $([math]::Round($l1l2l3l4Same/$seasonData.Count*100,2))% 的资源前4层相同
   - 仅靠L5（最后一层）区分不同季
   - L5承担了季度区分的主要责任

3. **L1L2L3相同率**: $([math]::Round($l1l2l3Same/$seasonData.Count*100,2))% 的资源前3层相同
   - L4和L5共同区分不同季

4. **L1L2相同率**: $([math]::Round($l1l2Same/$seasonData.Count*100,2))% 的资源前2层相同
   - 后3层(L3-L5)用于区分季度

5. **L1相同率**: $([math]::Round($l1Same/$seasonData.Count*100,2))% 的资源L1相同
   - 系列归属在顶层有一定体现

6. **完全不同**: $([math]::Round($allDifferent/$seasonData.Count*100,2))% 的资源连L1都不同
   - 大部分系列资源被完全分散

### L5层的作用

从L5分布可以看出：
- 如果L5分布极度不均（某些token占比很高），说明L5主要编码固定特征
- 如果L5分布较均匀，说明L5在做细粒度语义区分

### 实际意义

**优点**：
- 层级化编码保留了语义结构
- 前缀相同的资源在语义上有关联性

**缺点**：
- 5层完全相同的资源无法通过SID区分
- 大部分系列资源依然分散（$([math]::Round($allDifferent/$seasonData.Count*100,2))%完全不同）

### 对比任务3+L1 Sinkhorn

需要对比分析两个实验的差异：
- L1 Sinkhorn: 只在L1层用Sinkhorn
- L1+L4 Sinkhorn: 在L1和L4层用Sinkhorn

理论上L4 Sinkhorn应该改善L4-L5的利用率和区分度。

---

**生成时间**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")  
**数据来源**: $dataFile
"@

$report | Out-File -FilePath $outputFile -Encoding utf8
Write-Host "报告已生成: $outputFile"

