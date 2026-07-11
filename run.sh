#!/usr/bin/env bash
#
# Whispering 한번에 실행 스크립트
#
# 하는 일:
#   1. 시스템 의존성(python3-tk) 확인 및 설치
#   2. 가상환경(.venv) 생성 (없으면)
#   3. 앱과 의존성 설치 (최초 1회)
#   4. (GPU 사용 시) CUDA 라이브러리 설치 및 경로 설정
#   5. medium 모델을 ~/whispering-models 에 다운로드 (없으면)
#   6. Whispering 실행
#
# 사용법:
#   ./run.sh              # GPU(CUDA) 모드로 실행 (기본)
#   ./run.sh --cpu        # CPU 전용으로 실행 (CUDA 라이브러리 설치 생략)
#   MODEL=large-v3 ./run.sh   # 다른 모델 다운로드/사용

set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR=".venv"
MODELS_DIR="${MODELS_DIR:-$HOME/whispering-models}"
export WHISPERING_MODELS_DIR="$MODELS_DIR"
MODEL="${MODEL:-medium}"
MODEL_REPO="Systran/faster-whisper-${MODEL}"
MODEL_PATH="${MODELS_DIR}/faster-whisper-${MODEL}"
# Tell the app exactly which local model directory to use (no UI selection).
export WHISPERING_MODEL_PATH="$MODEL_PATH"

USE_GPU=1
if [[ "${1:-}" == "--cpu" ]]; then
    USE_GPU=0
fi

log() { printf '\n\033[1;36m[run.sh]\033[0m %s\n' "$*"; }

# 1) 시스템 의존성: tkinter (pip 로 설치 불가, OS 패키지 필요)
if ! python3 -c "import tkinter" >/dev/null 2>&1; then
    log "tkinter 미설치 감지 -> python3-tk 설치 (sudo 필요)"
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y python3-tk
    else
        echo "apt-get 이 없습니다. 배포판에 맞게 python3-tk(tkinter) 를 직접 설치하세요." >&2
        exit 1
    fi
fi

# 2) 가상환경 생성
if [[ ! -d "$VENV_DIR" ]]; then
    log "가상환경 생성: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 3) 앱 설치 (editable 설치로 항상 최신 소스 반영)
if ! python -c "import whispering, faster_whisper" >/dev/null 2>&1; then
    log "의존성 및 앱 설치"
    python -m pip install --upgrade pip
    python -m pip install -e .
    python -m pip install huggingface_hub
else
    python -m pip install -e . --no-deps --quiet
fi

# 4) GPU 모드: CUDA 라이브러리 설치 및 경로 설정
if [[ "$USE_GPU" == "1" ]]; then
    if ! python -c "import nvidia.cublas, nvidia.cudnn" >/dev/null 2>&1; then
        log "CUDA 라이브러리(cublas, cudnn) 설치"
        python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
    fi
    SITE_PKGS="$(python -c 'import site; print(site.getsitepackages()[0])')"
    for sub in cublas cudnn cuda_nvrtc; do
        libdir="$SITE_PKGS/nvidia/$sub/lib"
        [[ -d "$libdir" ]] && export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
    done
    log "CUDA 라이브러리 경로 설정 완료 (GPU 모드)"
else
    log "CPU 전용 모드"
fi

# 5) 모델 다운로드 (없으면)
if [[ ! -f "$MODEL_PATH/model.bin" ]]; then
    log "모델 다운로드: $MODEL_REPO -> $MODEL_PATH"
    mkdir -p "$MODELS_DIR"
    hf download "$MODEL_REPO" --local-dir "$MODEL_PATH"
else
    log "모델 이미 존재: $MODEL_PATH"
fi

# 6) 실행
log "Whispering 실행"
exec whispering
