# Script para download automatizado das bases Ninapro no Windows
# Requisito: PowerShell (Nativo no Windows 10/11)

# Definição das bases e seus padrões
$datasets = @(
    @{ Name = "DB1"; BaseUrl = "https://ninapro.hevs.ch/files/DB1/Preprocessed/s{0}.zip"; Range = 1..27 },
    @{ Name = "DB2"; BaseUrl = "https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s{0}.zip"; Range = 1..40 },
    @{ Name = "DB3"; BaseUrl = "https://ninapro.hevs.ch/files/db3_Preproc/s{0}_0.zip"; Range = 1..11 },
    @{ Name = "DB4"; BaseUrl = "https://ninapro.hevs.ch/files/DB4_Preproc/s{0}.zip"; Range = 1..10 },
    @{ Name = "DB5"; BaseUrl = "https://ninapro.hevs.ch/files/DB5_Preproc/s{0}.zip"; Range = 1..10 },
    @{ Name = "DB7"; BaseUrl = "https://ninapro.hevs.ch/files/DB7_Preproc/Subject_{0}.zip"; Range = 1..22 }
)

Write-Host "--- Iniciando Downloads Ninapro ---" -ForegroundColor Cyan

foreach ($db in $datasets) {
    $dirName = "Ninapro_" + $db.Name
    if (!(Test-Path $dirName)) {
        New-Item -ItemType Directory -Path $dirName | Out-Null
        Write-Host "Pasta criada: $dirName" -ForegroundColor Green
    }

    Write-Host "`nBaixando $($db.Name)..." -ForegroundColor Yellow

    foreach ($i in $db.Range) {
        $url = $db.BaseUrl -f $i
        $fileName = Split-Path $url -Leaf
        $destPath = Join-Path $dirName $fileName

        if (Test-Path $destPath) {
            Write-Host "Pulo: $fileName já existe." -ForegroundColor Gray
            continue
        }

        try {
            Write-Host "Baixando: $fileName ..." -NoNewline
            Invoke-WebRequest -Uri $url -OutFile $destPath -ErrorAction Stop
            Write-Host " [OK]" -ForegroundColor Green
        }
        catch {
            Write-Host " [FALHA]" -ForegroundColor Red
            Write-Warning "Não foi possível baixar $url"
        }
    }
}

Write-Host "`n--- Todos os downloads concluídos! ---" -ForegroundColor Cyan