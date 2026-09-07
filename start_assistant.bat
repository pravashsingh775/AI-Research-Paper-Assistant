@echo off
setlocal enabledelayedexpansion

title Lumen Research - AI Research Paper Assistant
color 0B

echo ========================================================================
echo                  LUMEN RESEARCH PLATFORM LAUNCHER                       
echo           Evidence-Backed AI Research Discovery and Analysis             
echo ========================================================================
echo.

:: 1. Navigate to project root directory
cd /d "%~dp0"
echo [*] Working directory: %CD%

:: 2. Ensure environment configuration exists
if not exist ".env" (
    echo [!] .env not found. Initializing from .env.example...
    copy ".env.example" ".env" >nul
    echo [OK] Created .env configuration file.
) else (
    echo [OK] Configuration file .env found.
)

:: 3. Verify Docker installation
where docker >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERROR] Docker was not found in your system PATH!
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    echo.
    pause
    exit /b 1
)

:: 4. Verify Docker daemon status and auto-launch Docker Desktop if needed
docker info >nul 2>&1
if %errorlevel% equ 0 goto docker_ready

echo.
echo [*] Docker daemon is currently stopped.
echo [*] Attempting to launch Docker Desktop automatically...

if exist "C:\Users\PRAVASH\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe" (
    start "" "C:\Users\PRAVASH\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe"
    goto wait_for_docker
)
if exist "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe" (
    start "" "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"
    goto wait_for_docker
)
if exist "C:\Program Files\Docker\Docker\Docker Desktop.exe" (
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    goto wait_for_docker
)

echo [!] Could not locate Docker Desktop executable automatically.
echo Please start Docker Desktop manually from the Start Menu, then press any key.
pause

:wait_for_docker
echo [*] Waiting for Docker engine to initialize [20 to 45 seconds]...
set /a attempts=0
set /a max_attempts=40

:check_docker_loop
ping 127.0.0.1 -n 4 >nul
set /a attempts+=1
docker info >nul 2>&1
if %errorlevel% equ 0 goto docker_ready

if !attempts! geq !max_attempts! (
    color 0C
    echo.
    echo [ERROR] Docker engine did not respond in time.
    echo Please ensure Docker Desktop is running and fully initialized, then run this file again.
    echo.
    pause
    exit /b 1
)
echo     ... waiting for Docker engine [Attempt !attempts! of !max_attempts!]...
goto check_docker_loop

:docker_ready
echo [OK] Docker engine is active and ready!

echo.
echo ========================================================================
echo [*] Starting multi-service stack via Docker Compose:
echo     - PostgreSQL 16 + pgvector (Vector Database)
echo     - Redis 7 (Job Queue)
echo     - MinIO Object Storage
echo     - Backend API (FastAPI + Alembic Migrations)
echo     - Background Worker (Document Extraction and Ingestion)
echo     - Frontend Web App (Next.js 15 UI)
echo ========================================================================
echo.

docker compose -f infra/docker/compose.yml up -d --build

if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERROR] Docker Compose failed to start the services.
    echo Check docker-compose logs or run: docker compose -f infra/docker/compose.yml logs
    echo.
    pause
    exit /b 1
)

echo.
echo [*] Waiting for backend API to report healthy...
set /a api_attempts=0

:wait_api
ping 127.0.0.1 -n 3 >nul
set /a api_attempts+=1
curl -s http://localhost:8000/health 2>nul | findstr "\"status\":\"ok\"" >nul 2>&1
if %errorlevel% equ 0 goto api_is_ready

if !api_attempts! geq 30 (
    echo [!] API health check timed out. Containers may still be compiling or migrating.
    goto open_browser
)
goto wait_api

:api_is_ready
echo [OK] Backend API is healthy!

:open_browser
echo.
echo [*] Launching Lumen Research in your default browser...
start http://localhost:3000

:dashboard
cls
color 0A
echo ========================================================================
echo               LUMEN RESEARCH PLATFORM IS NOW ONLINE                     
echo ========================================================================
echo.
echo   Application URLs:
echo     - Web Workspace [UI]:       http://localhost:3000
echo     - API Swagger Docs:         http://localhost:8000/docs
echo     - API Health Endpoint:      http://localhost:8000/health
echo     - MinIO Storage Console:    http://localhost:9001 (minio / miniosecret)
echo.
echo   Infrastructure:
echo     - PostgreSQL + pgvector:    localhost:5432 (DB: research, User: research)
echo     - Redis Job Queue:          localhost:6379 (Queue: research-paper-jobs)
echo.
echo ========================================================================
echo                           CONTROL MENU                                  
echo ========================================================================
echo   [1] Open Web Workspace in Browser (http://localhost:3000)
echo   [2] Open API Documentation in Browser (http://localhost:8000/docs)
echo   [3] Open MinIO Storage Console (http://localhost:9001)
echo   [4] View Live Container Logs (docker compose logs -f)
echo   [5] Run Integration Test Suite inside Backend
echo   [6] Ingest Local ArXiv Corpus Dataset (scripts/ingest_arxiv_csv.py)
echo   [7] Stop all services and shut down (docker compose down)
echo   [8] Exit launcher (leave services running in background)
echo ========================================================================
echo.
set /p choice="Select an option (1-8): "

if "%choice%"=="1" (
    start http://localhost:3000
    goto dashboard
)
if "%choice%"=="2" (
    start http://localhost:8000/docs
    goto dashboard
)
if "%choice%"=="3" (
    start http://localhost:9001
    goto dashboard
)
if "%choice%"=="4" (
    echo.
    echo [*] Press Ctrl+C at any time to return to the menu.
    echo.
    docker compose -f infra/docker/compose.yml logs -f
    goto dashboard
)
if "%choice%"=="5" (
    echo.
    echo [*] Running integration tests inside backend container...
    docker compose -f infra/docker/compose.yml exec backend python apps/api/tests/integration_e2e.py
    echo.
    pause
    goto dashboard
)
if "%choice%"=="6" (
    echo.
    echo [*] Ingesting ArXiv CSV to normalized JSONL...
    docker compose -f infra/docker/compose.yml exec backend python scripts/ingest_arxiv_csv.py data/raw/corpus/arxiv_scientific_dataset.csv data/processed/arxiv_papers.jsonl
    echo.
    pause
    goto dashboard
)
if "%choice%"=="7" (
    echo.
    echo [*] Stopping and removing all containers...
    docker compose -f infra/docker/compose.yml down
    echo [OK] All services stopped cleanly.
    ping 127.0.0.1 -n 3 >nul
    exit /b 0
)
if "%choice%"=="8" (
    echo.
    echo [*] Exiting launcher. Services will continue running in the background.
    ping 127.0.0.1 -n 3 >nul
    exit /b 0
)

echo [!] Invalid selection, please try again.
ping 127.0.0.1 -n 2 >nul
goto dashboard