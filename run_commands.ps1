Set-Location -Path "c:\Users\51183\Desktop\cigarette-RAG-QA\code"

$logDir = "..\logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force
}

$logFile = "$logDir\run_commands_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

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

$commands = @(
    @{Command = "python main.py --action prepare"; Description = "数据准备模式"},
    @{Command = "python main.py --action finetune"; Description = "RAG微调模式"},
    @{Command = "python main.py --action evaluate"; Description = "系统评估模式"},
    @{Command = "python main.py --action batch"; Description = "批量查询模式"}
)

Write-Log "开始执行命令序列" "INFO"
Write-Log "日志文件: $logFile" "INFO"

foreach ($cmd in $commands) {
    Write-Log "=====================================" "INFO"
    Write-Log "开始执行: $($cmd.Description)" "INFO"
    Write-Log "命令: $($cmd.Command)" "INFO"
    
    try {
        $output = & python $cmd.Command.Substring(7) 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Log "命令执行成功: $($cmd.Description)" "SUCCESS"
            $output | ForEach-Object { Write-Log $_ "OUTPUT" }
        } else {
            Write-Log "命令执行失败: $($cmd.Description)" "ERROR"
            Write-Log "退出码: