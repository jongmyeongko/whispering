# 윈도우 PC 설치 & 실행 가이드

이 문서는 새 **윈도우 PC** 에서 이 저장소를 내려받아 실행하는 방법을 처음부터 끝까지 설명합니다.

이 포크는 로컬 모델만 사용하도록 커스터마이징되어 있으며(모델 선택 UI 없음), `run.bat` 하나로 가상환경 생성 → 의존성 설치 → (GPU) CUDA 라이브러리 설치 → 모델 다운로드 → 실행까지 자동 처리합니다. 자세한 원본 대비 변경 내역은 [CHANGES.md](CHANGES.md) 를 참고하세요.

---

## 1. 사전 준비물

| 항목 | 필수 여부 | 설명 |
|------|-----------|------|
| **Python 3.11+** | 필수 | [python.org](https://www.python.org/downloads/) 에서 설치. 설치 시 **"Add python.exe to PATH"** 반드시 체크 |
| **인터넷 연결** | 필수(최초 1회) | 의존성/CUDA 라이브러리/모델 다운로드에 필요 |
| **NVIDIA GPU + 최신 드라이버** | 선택 | GPU(CUDA) 모드로 실행할 때. 없으면 `--cpu` 모드 사용 |
| **Git** | 선택 | `git clone` 으로 받을 때. 없으면 ZIP 다운로드로 대체 가능 |

> GPU 참고: GTX 1660 등 일부 지포스 카드는 순수 `float16` 이 느리거나 부정확할 수 있어, 이 앱의 기본값은 `int8_float16` 으로 설정되어 있습니다.

---

## 2. 소스 내려받기

### 방법 A) Git 사용 (권장)

명령 프롬프트(cmd) 또는 PowerShell 에서:

```bat
cd %USERPROFILE%
git clone https://github.com/jongmyeongko/whispering.git Whispering-main
cd Whispering-main
```

### 방법 B) ZIP 다운로드

1. https://github.com/jongmyeongko/whispering 접속
2. 초록색 **Code ▸ Download ZIP** 클릭
3. 원하는 위치(예: `C:\Users\<사용자>\Whispering-main`)에 압축 해제

---

## 3. 실행

압축을 푼(또는 clone 한) 폴더에서 `run.bat` 을 실행합니다.

### GPU (CUDA) 모드 — 기본

탐색기에서 **`run.bat` 더블클릭**, 또는 명령창에서:

```bat
run.bat
```

### CPU 전용 모드

NVIDIA GPU 가 없거나 CPU 로 돌리려면:

```bat
run.bat --cpu
```

### 다른 모델 사용

기본은 `medium` 입니다. 다른 크기를 쓰려면 환경변수 `MODEL` 을 지정합니다.

```bat
REM 명령 프롬프트(cmd)
set MODEL=small && run.bat
```

```powershell
# PowerShell
$env:MODEL="large-v3"; .\run.bat
```

사용 가능 값: `tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`, `large-v3`, `large`

### `run.bat` 이 자동으로 하는 일

1. Python 3.11+ 설치 확인
2. 가상환경 `.venv` 생성 (최초 1회)
3. 앱 + 의존성 설치 (최초 1회)
4. (GPU 모드) CUDA 12 / cuDNN 9 라이브러리 설치
5. 모델을 `%USERPROFILE%\whispering-models\faster-whisper-<모델>` 에 다운로드 (없을 때만)
6. Whispering GUI 실행

> 최초 실행은 의존성/모델 다운로드 때문에 시간이 걸립니다. 두 번째 실행부터는 빠르게 시작됩니다.

---

## 4. 프로그램 사용법

실행하면 GUI 가 뜹니다. 기본값은 다음과 같이 설정되어 있습니다.

- **Device**: `cuda`
- **Compute type**: `int8_float16`
- **Interval**: `3.0` (초 단위 처리 주기)

사용 순서:

1. **Mic** 에서 입력 장치 선택
   - 마이크 음성 → 마이크 장치 선택
   - PC 에서 재생되는 소리(유튜브/회의 등) 캡처 → **`[loopback]`** 이 붙은 장치 선택
2. 필요하면 **Source**(원문 언어), **Target**(번역 언어, `none` 이면 번역 안 함) 지정
3. (선택) **Auto-stop(min)** 에서 자동 종료 시간을 지정 (1분 단위, 최대 120분 / 기본값 `19` / `0` = 사용 안 함)
4. **Start** 클릭 → 왼쪽 패널에 전사, 오른쪽 패널에 번역이 실시간 표시
5. **Stop** 클릭으로 종료

### 자동 종료 예약 (Auto-stop)

- **Auto-stop(min)** 값을 1~120(1분 단위, 기본값 19)으로 지정하고 **Start** 를 누르면, 해당 시간이 지난 뒤 **자동으로 Stop** 됩니다.
- 실행 중에는 Stop 버튼 옆에 **남은 시간이 카운트다운**으로 표시됩니다. (예: `Auto-stop in 0:18:59`)
- `0` 이면 자동 종료를 사용하지 않습니다(수동 Stop).
- 자동 종료 시에도 전사 파일 저장(미확정 문장 포함)은 수동 Stop 과 동일하게 처리됩니다.

### 전사 결과 자동 저장

- **Start** 를 누르면 `%USERPROFILE%\whispering-transcripts\transcript_<날짜시간>.txt` 파일이 생성됩니다.
- 확정된 문장은 **실시간**으로 파일에 계속 저장되고, **Stop** 을 누르면 아직 확정되지 않은 문장까지 마지막에 저장됩니다.
- 실행 시작 시 콘솔 창에 저장 경로(`saving transcript to: ...`)가 출력됩니다.

### 문장 단위 줄바꿈 (후처리)

- 하단의 **`Split sentences`** 버튼을 누르고 저장된 전사 파일을 선택하면,
- 문장마다 한 줄로 정리된 **새 파일** `<원본이름>.sentences.txt` 가 생성됩니다. (원본은 그대로 유지)

---

## 5. 환경변수 (선택)

필요할 때만 사용합니다. cmd 는 `set NAME=VALUE`, PowerShell 은 `$env:NAME="VALUE"` 로 지정한 뒤 `run.bat` 을 실행하세요.

| 환경변수 | 기본값 | 설명 |
|----------|--------|------|
| `MODEL` | `medium` | 다운로드/사용할 모델 크기 |
| `MODELS_DIR` | `%USERPROFILE%\whispering-models` | 모델 저장 상위 폴더 |
| `WHISPERING_MODEL_PATH` | `<MODELS_DIR>\faster-whisper-<MODEL>` | 사용할 로컬 모델 폴더를 직접 지정 |
| `WHISPERING_TRANSCRIPT_DIR` | `%USERPROFILE%\whispering-transcripts` | 전사 파일 저장 폴더 |
| `WHISPERING_CAPTURE_BUFFER_SEC` | `30` | 오디오 캡처 버퍼(초). 클수록 드랍 방지에 유리 |

예) 이미 받아둔 모델 폴더를 그대로 쓰고 싶을 때:

```powershell
$env:WHISPERING_MODEL_PATH="D:\models\faster-whisper-medium"; .\run.bat
```

---

## 6. 문제 해결 (Troubleshooting)

### `Python not found on PATH`
Python 이 설치되지 않았거나 PATH 에 없습니다. [python.org](https://www.python.org/downloads/) 에서 3.11+ 를 설치하되 **"Add python.exe to PATH"** 를 체크하고 재설치하세요.

### `Library cublas64_12.dll is not found or cannot be loaded`
CUDA 라이브러리 로드 실패입니다. 이 포크는 이 문제를 자동 처리하도록 되어 있으니 대부분 다음으로 해결됩니다.
- 인터넷에 연결된 상태로 `run.bat` 을 다시 실행(CUDA 라이브러리 자동 설치)
- 여전히 실패하면 `.venv` 폴더를 삭제 후 `run.bat` 재실행
- GPU 가 없다면 `run.bat --cpu` 로 실행

### `Error 0x80070490` (오디오 장치 오류)
사용 가능한 입력 장치를 찾지 못한 경우입니다.
- 마이크가 연결/활성화되어 있는지 확인하거나,
- **Mic** 목록에서 `[loopback]` 장치를 선택하세요. (물리 마이크가 없어도 시스템 소리 캡처 가능)

### `[whispering] AUDIO DROP #...` 로그
실제로 입력 오디오가 유실되는 상황입니다(전사가 실시간을 못 따라감).
- **Interval** 을 늘리거나(예: 3.0), 더 작은 **모델**(`small`) 또는 더 가벼운 **Compute type**(`int8`) 사용
- 또는 `WHISPERING_CAPTURE_BUFFER_SEC` 값을 키워 버퍼를 늘리세요.

### `[whispering] latency ~Ns behind live (RTF=...)` 로그
오디오가 버퍼링되어 **지연**되고 있을 뿐, **유실은 아닙니다**. 잠깐씩 나오는 것은 정상입니다. 지속적으로 지연이 커지면 위 AUDIO DROP 항목과 동일하게 조치하세요.

### 종료
GUI 창을 닫으면 콘솔도 바로 종료됩니다. (설치/다운로드 실패 시에만 오류 확인을 위해 멈춥니다.)

---

## 7. 요약 (Quick Start)

```bat
git clone https://github.com/jongmyeongko/whispering.git Whispering-main
cd Whispering-main
run.bat
```

GPU 가 없으면 `run.bat --cpu` 로 실행하면 됩니다.
