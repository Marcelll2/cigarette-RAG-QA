#!/usr/bin/env powershell

# 简单运行脚本 - 顺序执行命令，遇到错误时收集信息并继续

# 设置工作目录
Set-Location -Path "c:\Users\51183\Desktop\cigarette-RAG-QA\code"

# 创建日志目录
$logDir = "..\logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force
}

# 日志文件
$logFile = "$logDir\run_simple_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

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
    Write-Log "开始执行: $($cmd.Description)" "INFO"
    Write-Log "命令: $($cmd.Command)" "INFO"
    
    # 执行命令
    $process = Start-Process -FilePath "python" -ArgumentList $cmd.Command.Substring(7) -NoNewWindow -PassThru -RedirectStandardOutput "temp_output.txt" -RedirectStandardError "temp_error.txt"
    
    # 等待命令完成
    $process.WaitForExit()
    
    # 读取输出
    $stdout = Get-Content "temp_output.txt" -ErrorAction SilentlyContinue
    $stderr = Get-Content "temp_error.txt" -ErrorAction SilentlyContinue
    
    # 检查执行结果
    if ($process.ExitCode -eq 0) {
        Write-Log "命令执行成功: $($cmd.Description)" "SUCCESS"
        $stdout | ForEach-Object { Write-Log $_ "OUTPUT" }
    } else {
        Write-Log "命令执行失败: $($cmd.Description)" "ERROR"
        Write-Log "退出码: $($process.ExitCode)" "ERROR"
        if ($stdout) {
            Write-Log "标准输出: $stdout" "ERROR"
        }
        if ($stderr) {
            Write-Log "错误输出: $stderr" "ERROR"
        }
    }
    
    # 清理临时文件
    Remove-Item "temp_output.txt" -ErrorAction SilentlyContinue
    Remove-Item "temp_error.txt" -ErrorAction SilentlyContinue
    
    Write-Log "=====================================" "INFO"
    Write-Log "" "INFO"
}

Write-Log "命令序列执行完成" "INFO"
Write-Log "请查看日志文件获取详细信息: $logFile" "INFO"
