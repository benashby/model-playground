@echo off
rem Build llama-server from a pinned llama.cpp tag on native Windows, the counterpart of
rem build_llama_cpp.sh. Needs Git, CMake, Ninja, Visual Studio's C++ tools and, for cuda,
rem the CUDA toolkit.
rem
rem   models\nimble-tev1\probes\build_llama_cpp.cmd cuda          rem the GPU half: CUDA 13.4, sm_86 (RTX 3090)
rem   models\nimble-tev1\probes\build_llama_cpp.cmd cpu
rem   models\nimble-tev1\probes\build_llama_cpp.cmd cpu b9190     rem any tag
rem
rem Builds outside the repository, in %LLAMA_BUILD_ROOT% (default %USERPROFILE%\.cache\llama-builds),
rem and prints the binary path and its --version line last. Optional variables:
rem   VCVARS     vcvars64.bat to load (default: Visual Studio 2026 Community)
rem   CUDA_PATH  toolkit to build against (the machine default otherwise)
rem   CUDA_ARCH  CMAKE_CUDA_ARCHITECTURES (default 86)
setlocal
set BACKEND=%1
set TAG=%2
if "%BACKEND%"=="" (echo usage: build_llama_cpp.cmd cuda^|cpu [tag] & exit /b 2)
if "%TAG%"=="" set TAG=b11232
if "%LLAMA_BUILD_ROOT%"=="" set LLAMA_BUILD_ROOT=%USERPROFILE%\.cache\llama-builds
if "%VCVARS%"=="" set VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat
if "%CUDA_ARCH%"=="" set CUDA_ARCH=86
set SRC=%LLAMA_BUILD_ROOT%\llama.cpp-%TAG%
set BUILD=%SRC%\build-%BACKEND%
if "%BACKEND%"=="cuda" (set FLAGS=-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH%) else if "%BACKEND%"=="cpu" (set FLAGS=) else (echo unknown backend %BACKEND% & exit /b 2)

call "%VCVARS%" >nul || exit /b 1
if not exist "%LLAMA_BUILD_ROOT%" mkdir "%LLAMA_BUILD_ROOT%"
if not exist "%SRC%\.git" git clone --quiet --depth 1 --branch %TAG% https://github.com/ggml-org/llama.cpp "%SRC%" || exit /b 1
for /f %%h in ('git -C "%SRC%" rev-parse HEAD') do echo llama.cpp %TAG% at %%h 1>&2
cmake -S "%SRC%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF %FLAGS% 1>&2 || exit /b 1
cmake --build "%BUILD%" --target llama-server -j %NUMBER_OF_PROCESSORS% 1>&2 || exit /b 1
echo %BUILD%\bin\llama-server.exe
"%BUILD%\bin\llama-server.exe" --version 2>&1 | findstr /B /C:"version" /C:"built with"
