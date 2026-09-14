# 验证L5 token与saletype的映射关系
$dataFile = "results_three_way\exp_task3_l1_sinkhorn\indices\indices.jsonl"
$outputFile = "L5_token_saletype_映射验证.md"

Write-Host "开始分析L5 token与saletype的映射关系..."

$l5ToSaletype = @{}
$saletypeToL5 = @{}
$allSaletypes = @{}
$totalLines = 0
$hasSaletypeLines = 0

Get-Content $dataFile -Encoding utf8 | ForEach-Object {
    $totalLines++
    if ($totalLines % 50000 -eq 0) {
        Write-Host "  已处理 $totalLines 行..."
    }
    
    $d = $_ | ConvertFrom-Json
    
    if ($d.PSObject.Properties['saletype'] -and $d.tokens.Count -ge 5) {
        $hasSaletypeLines++
        $l5Token = $d.tokens[4]
        $saletype = $d.saletype
        
        # 统计每个L5 token对应的saletype
        if (-not $l5ToSaletype.ContainsKey($l5Token)) {
            $l5ToSaletype[$l5Token] = @{}
        }
        if (-not $l5ToSaletype[$l5Token].ContainsKey($saletype)) {
            $l5ToSaletype[$l5Token][$saletype] = 0
        }
        $l5ToSaletype[$l5Token][$saletype]++
        
        # 统计每个saletype对应的L5 token
        if (-not $saletypeToL5.ContainsKey($saletype)) {
            $saletypeToL5[$saletype] = @{}
        }
        if (-not $saletypeToL5[$saletype].ContainsKey($l5Token)) {
            $saletypeToL5[$saletype][$l5Token] = 0
        }
        $saletypeToL5[$saletype][$l5Token]++
        
        # 统计总的saletype
        if (-not $allSaletypes.ContainsKey($saletype)) {
            $allSaletypes[$saletype] = 0
        }
        $allSaletypes[$saletype]++
    }
}

Write-Host "`n数据加载完成，开始分析..."

# 分析一对一、一对多、多对一映射
$oneToOne = 0
$oneToMany = 0
$manyToOne = 0
$conflictTokens = @()

foreach ($token in $l5ToSaletype.Keys) {
    $numSaletypes = $l5ToSaletype[$token].Count
    if ($numSaletypes -eq 1) {
        $oneToOne++
    } else {
        $oneToMany++
        $conflictTokens += @{
            token = $token
            saletypes = $l5ToSaletype[$token].Keys
            counts = $l5ToSaletype[$token]
        }
    }
}

# 生成报告
$report = @"
# L5 Token 与 Saletype 映射关系验证报告

---

## 📊 基本统计

| 指标 | 数值 |
|---|---:|
| **总数据量** | $totalLines |
| **有saletype的数据** | $hasSaletypeLines |
| **不同的saletype种类** | $($allSaletypes.Count) |
| **使用的L5 token数** | $($l5ToSaletype.Count) |
| **平均每个token对应的saletype种类** | $([math]::Round($allSaletypes.Count / $l5ToSaletype.Count, 2)) |

---

## 🎯 核心发现

### 映射关系统计

| 映射类型 | 数量 | 比例 | 说明 |
|---|---:|---:|---|
| **一对一** (1 token → 1 saletype) | $oneToOne | $([math]::Round($oneToOne/$l5ToSaletype.Count*100,2))% | ✅ 理想映射 |
| **一对多** (1 token → 多个 saletype) | $oneToMany | $([math]::Round($oneToMany/$l5ToSaletype.Count*100,2))% | ⚠️ 存在冲突 |

---

## 📋 所有Saletype列表

| Saletype | 出现次数 | 占比 | 主要使用的L5 Token |
|---|---:|---:|---|
"@

# 按出现次数排序saletype
$sortedSaletypes = $allSaletypes.GetEnumerator() | Sort-Object -Property Value -Descending

foreach ($st in $sortedSaletypes) {
    $saletype = $st.Key
    $count = $st.Value
    $pct = [math]::Round($count / $hasSaletypeLines * 100, 2)
    
    # 找出这个saletype最常用的L5 token
    $topTokens = $saletypeToL5[$saletype].GetEnumerator() | 
                 Sort-Object -Property Value -Descending | 
                 Select-Object -First 3
    $tokenStr = ($topTokens | ForEach-Object { "$($_.Key)($($_.Value))" }) -join ", "
    
    $report += "| ``$saletype`` | $count | $pct% | $tokenStr |`n"
}

$report += @"

---

## ⚠️ 存在冲突的L5 Token (一对多映射)

"@

if ($oneToMany -gt 0) {
    $report += @"

共有 **$oneToMany** 个L5 token对应多个saletype，说明存在映射冲突！

| L5 Token | 对应的Saletype种类数 | Saletype列表 | 数量分布 |
|---|---:|---|---|
"@
    
    $conflictTokensSorted = $conflictTokens | Sort-Object { $_.saletypes.Count } -Descending | Select-Object -First 20
    
    foreach ($item in $conflictTokensSorted) {
        $token = $item.token
        $saletypes = $item.saletypes -join ", "
        $counts = ($item.counts.GetEnumerator() | ForEach-Object { "$($_.Key):$($_.Value)" }) -join ", "
        $numSaletypes = $item.saletypes.Count
        
        $report += "| ``$token`` | $numSaletypes | ``$saletypes`` | $counts |`n"
    }
    
    $report += @"

### 冲突分析

- **冲突率**: $([math]::Round($oneToMany/$l5ToSaletype.Count*100,2))%
- **冲突原因可能**:
  1. 训练不充分，模型未完全学习到saletype的区分特征
  2. L5层被其他特征干扰，不是纯粹的saletype表示
  3. 某些saletype本身就很相似，模型难以区分

"@
} else {
    $report += @"

✅ **没有冲突！** 所有L5 token都是一对一映射到saletype。

这说明：
- 模型训练充分
- L5层确实被成功hardcode为saletype
- 表征空间充足

"@
}

$report += @"

---

## 💡 关键结论

### 结论1：Saletype种类数

实际有 **$($allSaletypes.Count)** 种不同的saletype。

"@

if ($allSaletypes.Count -le 256) {
    $report += @"

✅ **256个token充足！**

- 实际需要: $($allSaletypes.Count)种
- 可用空间: 256个token
- 冗余空间: $(256 - $allSaletypes.Count)个token ($([math]::Round((256 - $allSaletypes.Count)/256*100,2))%)

"@
} else {
    $report += @"

❌ **256个token不足！**

- 实际需要: $($allSaletypes.Count)种
- 可用空间: 256个token
- 缺口: $($allSaletypes.Count - 256)个token

**建议**: 增大L5 codebook到至少 $([math]::Ceiling($allSaletypes.Count * 1.2)) 个token。

"@
}

$report += @"

### 结论2：当前使用$($l5ToSaletype.Count)个L5 token的原因

"@

if ($allSaletypes.Count -eq $l5ToSaletype.Count -and $oneToMany -eq 0) {
    $report += @"

**原因**: 每种saletype对应一个唯一的L5 token，完美映射！

- $($allSaletypes.Count)种saletype → $($l5ToSaletype.Count)个L5 token
- 一对一映射，无冲突
- 说明模型训练得很好

"@
} elseif ($allSaletypes.Count -lt $l5ToSaletype.Count) {
    $report += @"

**原因**: 某些saletype被映射到了多个L5 token（多对一）

- $($allSaletypes.Count)种saletype → $($l5ToSaletype.Count)个L5 token
- 平均每种saletype用了 $([math]::Round($l5ToSaletype.Count / $allSaletypes.Count, 2)) 个token
- 说明训练不充分，存在冗余映射

"@
} else {
    $report += @"

**原因**: 某些L5 token对应多个saletype（一对多冲突）

- $($allSaletypes.Count)种saletype → $($l5ToSaletype.Count)个L5 token
- 存在 $oneToMany 个冲突token
- 说明训练不充分或特征纠缠

"@
}

$report += @"

### 结论3：256个token是否充足？

"@

if ($allSaletypes.Count -le 256) {
    $redundancy = 256 - $allSaletypes.Count
    $redundancyPct = [math]::Round($redundancy / 256 * 100, 2)
    
    $report += @"

✅ **充足！**

| 指标 | 数值 |
|---|---:|
| 实际saletype种类 | $($allSaletypes.Count) |
| L5=256时可用空间 | 256 |
| 冗余空间 | $redundancy ($redundancyPct%) |

**建议**:
"@
    
    if ($redundancyPct -gt 50) {
        $report += @"

- 冗余空间充足（>50%），可以考虑减小codebook到 $([math]::Ceiling($allSaletypes.Count * 1.5)) 提升效率
- 或保持256以应对未来扩展
"@
    } elseif ($redundancyPct -gt 20) {
        $report += @"

- 冗余空间适中（20-50%），256是合理的选择
- 可以应对一定的业务扩展
"@
    } else {
        $report += @"

- ⚠️ 冗余空间偏小（<20%），建议增大到512以提供更多扩展空间
- 或者优化业务逻辑，减少saletype种类
"@
    }
} else {
    $shortage = $allSaletypes.Count - 256
    $report += @"

❌ **不足！**

| 指标 | 数值 |
|---|---:|
| 实际saletype种类 | $($allSaletypes.Count) |
| L5=256时可用空间 | 256 |
| 缺口 | $shortage |

**建议**:
- 必须增大L5 codebook到至少 $([math]::Ceiling($allSaletypes.Count * 1.2))
- 或者简化业务逻辑，减少saletype种类到256以内
"@
}

$report += @"

---

## 📊 Saletype分布可视化

### Top 10 Saletype

``````
"@

$topSaletypes = $sortedSaletypes | Select-Object -First 10
$maxCount = $topSaletypes[0].Value
foreach ($st in $topSaletypes) {
    $saletype = $st.Key
    $count = $st.Value
    $pct = [math]::Round($count / $hasSaletypeLines * 100, 2)
    $barLength = [math]::Floor($count / $maxCount * 50)
    $bar = "█" * $barLength
    
    $report += "$saletype`.PadRight(20) | $bar $count ($pct%)`n"
}

$report += @"
``````

---

**报告生成时间**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")  
**数据来源**: $dataFile  
**分析行数**: $totalLines 行  
**有效数据**: $hasSaletypeLines 行
"@

$report | Out-File -FilePath $outputFile -Encoding utf8
Write-Host "`n✅ 报告已生成: $outputFile"
