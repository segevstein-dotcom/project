@echo off
REM Build script for LayerNorm C implementation
echo Compiling LayerNorm...
gcc -o layernorm_test.exe main.c utils.c -lm -Wall
if %ERRORLEVEL% EQU 0 (
    echo Build successful! Executable: src/layernorm_test.exe
) else (
    echo Build failed!
    exit /b 1
)
