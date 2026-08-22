# push_to_main.ps1
# One-click Git branch fix and push script

$ErrorActionPreference = "Stop"
$ProjectPath = "D:\aistudypython\pyharness"

Write-Host "Starting Git branch fix and push script..." -ForegroundColor Cyan

# 1. Enter project directory
Write-Host "[1/7] Changing directory to $ProjectPath ..." -ForegroundColor Yellow
if (Test-Path $ProjectPath) {
    Set-Location $ProjectPath
} else {
    Write-Host "ERROR: Cannot find directory $ProjectPath" -ForegroundColor Red
    exit 1
}

# 2. Check and rename branch
Write-Host "[2/7] Checking current branch..." -ForegroundColor Yellow
$currentBranch = git branch --show-current
if ($currentBranch -eq "master") {
    Write-Host "  -> Current is master, renaming to main..." -ForegroundColor Green
    git branch -m master main
} elseif ($currentBranch -eq "main") {
    Write-Host "  -> Already on main, skipping rename." -ForegroundColor Green
} else {
    Write-Host "  -> Current branch is $currentBranch, switching to main..." -ForegroundColor Green
    git checkout main
}

# 3. Stage changes
Write-Host "[3/7] Staging all changes (git add .)..." -ForegroundColor Yellow
git add .

# 4. Commit changes
Write-Host "[4/7] Committing changes..." -ForegroundColor Yellow
git commit -m "chore: sync branch to main and fix CI"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  -> No new changes to commit, continuing to push..." -ForegroundColor DarkYellow
}

# 5. Push to main and set upstream
Write-Host "[5/7] Pushing code to remote main branch..." -ForegroundColor Yellow
Write-Host "  WARNING: Please enter your GitHub Username and Token when prompted!" -ForegroundColor Red
git push -u origin main
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Push failed! Please check your network or Token permissions." -ForegroundColor Red
    exit 1
}

# 6. Try to delete remote master branch
Write-Host "[6/7] Trying to clean up remote master branch..." -ForegroundColor Yellow
git push origin --delete master
if ($LASTEXITCODE -ne 0) {
    Write-Host "  -> No remote master branch to clean up." -ForegroundColor DarkYellow
    $global:LASTEXITCODE = 0
} else {
    Write-Host "  -> Successfully deleted remote master branch!" -ForegroundColor Green
}

# 7. Set global default branch
Write-Host "[7/7] Setting Git global default initial branch to main..." -ForegroundColor Yellow
git config --global init.defaultBranch main

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "DONE! Your code has been successfully pushed to main!" -ForegroundColor Green
Write-Host "Please go to GitHub Actions page to check CI status." -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Cyan