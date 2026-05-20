@echo off
title Iniciar Totem LSB - Universidad Católica Boliviana
echo =====================================================================
echo   Iniciando Servidor Local de IA - Feria de Carreras 2025 (Tótem UCB)
echo =====================================================================
echo.

:: 1. Verificar si existe el entorno virtual
if not exist ".venv312\Scripts\activate.bat" (
    echo [ERROR] No se encontró el entorno virtual .venv312.
    echo Por favor copia la carpeta completa con el entorno virtual.
    pause
    exit /b
)

:: 2. Activar el entorno virtual
call .venv312\Scripts\activate.bat

:: 3. Iniciar el servidor Flask
echo [1/2] Cargando modelos de Inteligencia Artificial y levantando servidor...
start /b python app.py

:: 4. Esperar a que se carguen los modelos (5 segundos)
echo Esperando inicialización...
timeout /t 5 /nobreak > nul

:: 5. Lanzar el navegador en Modo Kiosko
echo [2/2] Abriendo interfaz en pantalla completa...
:: Intentar con Google Chrome
where chrome.exe >nul 2>nul
if %errorlevel% equ 0 (
    start "" chrome.exe --kiosk --new-window http://127.0.0.1:5000
) else (
    :: Si no tiene Chrome, intentar con Microsoft Edge en modo kiosko
    start "" msedge.exe --kiosk http://127.0.0.1:5000 --edge-kiosk-type=fullscreen
)

echo.
echo =====================================================================
echo   El Tótem se está ejecutando correctamente en localhost:5000.
echo   Presiona CTRL+C en esta ventana para detener el servidor de IA.
echo =====================================================================
