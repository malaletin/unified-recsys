<#
    Оркестратор Трека 3 (Windows PowerShell).

    Запускает полное сравнение моделей, ablation, подбор гиперпараметров и
    строит график. Перед запуском активируйте окружение:
        .\.venv\Scripts\Activate.ps1

    Примеры:
        .\run_track3.ps1 -Data data\trivago
        .\run_track3.ps1 -Data data\trivago -Epochs 10
        .\run_track3.ps1 -Data data\trivago_sample -Epochs 3   # быстрая проверка
#>
param(
    [string]$Data = "data\trivago",
    [int]$Epochs = 10,
    [int]$Trials = 30
)

$ErrorActionPreference = "Stop"

function Step($title) {
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Cyan
    Write-Host " $title" -ForegroundColor Cyan
    Write-Host "==================================================================" -ForegroundColor Cyan
}

Step "Проверка GPU (PyTorch + CUDA)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Step "Сравнение всех моделей (data=$Data, epochs=$Epochs)"
python -m scripts.compare_models --data $Data --epochs $Epochs

Step "График сравнения (results\track3_mrr.png)"
python -m scripts.plot_track3

Step "Ablation признаков LightGBM"
python -m scripts.ablation_ltr --data $Data

Step "Подбор гиперпараметров LightGBM (Optuna, trials=$Trials)"
python -m scripts.tune_optuna --data $Data --trials $Trials

Step "Готово"
Write-Host "Результаты:" -ForegroundColor Green
Write-Host "  results\track3_comparison.json" -ForegroundColor Green
Write-Host "  results\track3_mrr.png" -ForegroundColor Green
Write-Host "  results\ltr_ablation.json" -ForegroundColor Green
Write-Host "  results\ltr_best_params.json" -ForegroundColor Green
