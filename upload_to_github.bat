@echo off
echo ============================================================
echo GitHub Upload for ZLjjj/rqvae-reproduction
echo ============================================================
echo.
echo STEP 1: Create GitHub repository
echo.
echo Please visit: https://github.com/new
echo.
echo Fill in:
echo   Repository name: rqvae-reproduction
echo   Description: RQVAE + L1+L4 Sinkhorn (CR=0.0907%%)
echo   Public
echo   DO NOT check "Add a README"
echo.
echo Press Enter after you created the repository...
pause
echo.
echo STEP 2: Pushing code...
cd /d C:\Users\dszlj\Desktop\gensearchrec-main-0901\github_upload
git config user.name "ZLjjj"
git config user.email "ZLjjj@users.noreply.github.com"
git remote add origin https://github.com/ZLjjj/rqvae-reproduction.git 2>nul
git branch -M main
git push -u origin main
echo.
echo ============================================================
echo Done! Visit: https://github.com/ZLjjj/rqvae-reproduction
echo ============================================================
pause

