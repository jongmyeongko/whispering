#!/usr/bin/env bash
#
# WSL2 -> Windows 배포 스크립트
#
# WSL2 에서 이 프로젝트를 윈도우 폴더(/mnt/c/... 또는 C:\...)로 복사합니다.
#   - 개발 아티팩트(.venv, __pycache__, .git 등)는 제외
#   - .bat 파일은 윈도우용 CRLF 개행으로 변환
#   - (옵션) WSL 에 받아둔 모델을 윈도우 사용자 폴더로 함께 복사
#
# 사용법:
#   ./deploy-to-windows.sh <윈도우_대상경로>
#   ./deploy-to-windows.sh 'C:\Whispering'
#   ./deploy-to-windows.sh /mnt/c/Users/jake/Whispering
#   ./deploy-to-windows.sh --with-model 'C:\Whispering'   # 모델도 함께 복사
#
# 대상경로 생략 시: <윈도우 사용자 홈>\Whispering-main 으로 배포합니다.

set -euo pipefail
cd "$(dirname "$0")"

SRC="$(pwd)"

log()  { printf '\n\033[1;36m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[deploy] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[deploy] %s\033[0m\n' "$*" >&2; exit 1; }

# --- 인자 파싱 ---
WITH_MODEL=0
DEST_ARG=""
for arg in "$@"; do
    case "$arg" in
        --with-model) WITH_MODEL=1 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) DEST_ARG="$arg" ;;
    esac
done

# --- 윈도우 사용자 홈 탐지 (/mnt/c/Users/<user>) ---
# 주의: cmd.exe 는 WSL 경로에서 실행하면 실패하므로 /mnt/c 에서 실행한다.
win_home=""
if command -v cmd.exe >/dev/null 2>&1 && command -v wslpath >/dev/null 2>&1; then
    wp="$( (cd /mnt/c 2>/dev/null && cmd.exe /c "echo %USERPROFILE%" 2>/dev/null) | tr -d '\r\n' || true)"
    if [[ -n "$wp" ]]; then
        win_home="$(wslpath -u "$wp" 2>/dev/null || true)"
    fi
fi

# --- 대상 경로 결정 및 WSL 경로로 변환 ---
if [[ -z "$DEST_ARG" ]]; then
    [[ -n "$win_home" ]] || die "대상 경로를 지정하세요. 예: ./deploy-to-windows.sh 'C:\\Whispering'"
    DEST="$win_home/Whispering-main"
else
    if [[ "$DEST_ARG" =~ ^[A-Za-z]:[\\/] || "$DEST_ARG" == *"\\"* ]]; then
        command -v wslpath >/dev/null 2>&1 || die "wslpath 가 없어 윈도우 경로를 변환할 수 없습니다."
        DEST="$(wslpath -u "$DEST_ARG")"
    else
        DEST="$DEST_ARG"
    fi
fi

case "$DEST" in
    /mnt/*) : ;;
    *) warn "대상 '$DEST' 이(가) 윈도우 마운트(/mnt/...) 경로가 아닌 것 같습니다. 계속 진행합니다." ;;
esac

log "복사 원본:  $SRC"
log "복사 대상:  $DEST"
mkdir -p "$DEST"

# --- 제외 목록 ---
EXCLUDES=(
    ".venv" "venv" "__pycache__" ".git" ".github"
    "build" "dist" "*.egg-info" ".pytest_cache"
    "*.pyc" "*.pyo" ".mypy_cache" ".ruff_cache" ".vscode"
    "*:Zone.Identifier"
)

# --- 복사 (rsync 우선, 없으면 tar 파이프) ---
if command -v rsync >/dev/null 2>&1; then
    rsync_excludes=()
    for e in "${EXCLUDES[@]}"; do rsync_excludes+=(--exclude "$e"); done
    rsync -a --delete "${rsync_excludes[@]}" "$SRC"/ "$DEST"/
else
    warn "rsync 미설치 -> tar 로 복사합니다(대상 기존 파일은 유지)."
    tar_excludes=()
    for e in "${EXCLUDES[@]}"; do tar_excludes+=(--exclude="$e"); done
    tar -C "$SRC" "${tar_excludes[@]}" -cf - . | tar -C "$DEST" -xf -
fi

# --- .bat 파일 CRLF 변환 ---
log "윈도우 배치 파일(.bat) 개행을 CRLF 로 변환"
while IFS= read -r -d '' bat; do
    sed -i 's/\r$//; s/$/\r/' "$bat"
done < <(find "$DEST" -maxdepth 2 -name '*.bat' -print0)

# --- (옵션) 모델 복사 ---
if [[ "$WITH_MODEL" == "1" ]]; then
    src_models="${WHISPERING_MODELS_DIR:-$HOME/whispering-models}"
    if [[ -d "$src_models" ]]; then
        if [[ -n "$win_home" ]]; then
            dst_models="$win_home/whispering-models"
            log "모델 복사: $src_models -> $dst_models"
            mkdir -p "$dst_models"
            if command -v rsync >/dev/null 2>&1; then
                rsync -a "$src_models"/ "$dst_models"/
            else
                cp -r "$src_models"/. "$dst_models"/
            fi
        else
            warn "윈도우 사용자 홈을 찾지 못해 모델 복사를 건너뜁니다. (윈도우에서 run.bat 실행 시 자동 다운로드됩니다.)"
        fi
    else
        warn "복사할 모델이 없습니다: $src_models (윈도우에서 run.bat 실행 시 자동 다운로드됩니다.)"
    fi
fi

log "배포 완료!"
echo
echo "다음 단계 (윈도우에서):"
echo "  1) 탐색기에서 대상 폴더로 이동"
echo "  2) run.bat 더블클릭 (또는 명령창에서 run.bat)"
echo
if [[ -n "$win_home" ]]; then
    win_dest="$(command -v wslpath >/dev/null 2>&1 && wslpath -w "$DEST" 2>/dev/null || echo "$DEST")"
    echo "  윈도우 경로: $win_dest"
fi
