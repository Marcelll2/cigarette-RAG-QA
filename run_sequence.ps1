#!/usr/bin/env powershell

# 运行顺序脚本 - 顺序执行命令，遇到错误时收集信息并继续

# 设置工作目录
Set-Location -Path "c:\Users\51183\Desktop\cigarette-RAG-QA\code"

# 创建日志目录
$logDir = "..\logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force
}

# 日志文件
$logFile = "$logDir\run_sequence_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# 写入日志函数
function Write-Log {
    param (
        [string]$Message,
        [string]$Level = "INFO"
    )
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $logEntry = "[$timestamp] [$Level] $Message"
    Write-Host $logEntry
    Add-Content -Path $logFile -Value $logEntry
}

# 运行命令函数
function Run-Command {
    param (
        [string]$Command,
        [string]$Description
    )
    
    Write-Log "开始执行: $Description" "INFO"
    Write-Log "命令: $Command" "INFO"
    
    try {
        # 执行命令并捕获输出
        $output = & cmd.exe /c "$Command" 2>&1
        
        # 检查是否有错误
        if ($LASTEXITCODE -ne 0) {
            throw "命令执行失败，退出码: $LASTEXITCODE"
        }
        
        # 写入输出到日志
        $output | ForEach-Object { Write-Log $_ "OUTPUT" }
        Write-Log "命令执行成功: $Description" "SUCCESS"
        return $true
    }
    catch {
        Write-Log "命令执行失败: $Description" "ERROR"
        Write-Log "错误信息: $($_.Exception.Message)" "ERROR"
        Write-Log "错误详情: $output" "ERROR"
        return $false
    }
}

# 要执行的命令列表
$commands = @(
    @{
        Command = "python main.py --action prepare"
        Description = "数据准备模式"
    },
    @{
        Command = "python main.py --action finetune"
        Description = "RAG微调模式"
    },
    @{
        Command = "python main.py --action evaluate"
        Description = "系统评估模式"
    },
    @{
        Command = "python main.py --action batch"
        Description = "批量查询模式"
    }
)

# 开始执行
Write-Log "开始顺序执行命令序列" "INFO"
Write-Log "日志文件: $logFile" "INFO"

# 执行每个命令
foreach ($cmd in $commands) {
    Write-Log "=====================================" "INFO"
    Run-Command -Command $cmd.Command -Description $cmd.Description
    Write-Log "=====================================" "INFO"
    Write-Log "" "INFO"
}

Write-Log "命令序列执行完成" "INFO"
Write-Log "请查看日志文件获取详细信息: $logFile" "INFO"
