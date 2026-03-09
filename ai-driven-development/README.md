# AI DJ Assist

프로 DJ를 위한 AI 기반 데스크톱 어시스턴트입니다. Rekordbox, Serato, Traktor 등 기존 DJ 소프트웨어와 함께 실행되며, 실시간 트랙 추천, 하모닉 믹싱 제안, AI 블렌드 프리뷰를 제공합니다.

공연 중 BPM, 키, 에너지, 장르, 관객 반응을 동시에 고려해야 하는 인지 부하를 줄여주면서도 DJ의 창작 자유를 존중합니다.

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| 데스크톱 앱 | Tauri v2 + React + TypeScript |
| 오디오 분석 & ML | Python 3.11+ (Essentia, PyTorch, ONNX) |
| DSP & 오디오 출력 | Rust (cpal, midir) |
| 커뮤니티 백엔드 | FastAPI + PostgreSQL |
| 결제 | Stripe (Pro 월간 / 연간) |

---

## 시작하기

### 사전 준비

시작하기 전에 아래 도구들이 설치되어 있어야 합니다:

- **Node.js 20+** - [nodejs.org](https://nodejs.org)에서 다운로드
- **Python 3.11 ~ 3.12** - [python.org](https://www.python.org)에서 다운로드 (3.13은 essentia 미지원)
- **Rust (stable)** - 터미널에서 `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` 실행 후 `source $HOME/.cargo/env`
- **PostgreSQL** - 백엔드 사용 시에만 필요

### 설치 및 실행

```bash
# 1. 저장소 클론
git clone <repository-url>
cd ai-driven-development

# 2. Node 의존성 설치
npm install

# 3. Python 분석 패키지 설정
cd packages/analysis
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,audio,ml]"
cd ../..

# 4. Rust DSP 빌드 (Rust 설치 필수)
cd packages/dsp
cargo build                        # 또는 rustup run stable cargo build
cd ../..

# 5. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어서 필요한 값을 입력하세요

# 6. 프론트엔드만 실행 (Rust 없이도 가능)
npm run dev                        # Vite 개발 서버 → http://localhost:1420

# 6-1. Tauri 데스크톱 앱 실행 (Rust 필요)
cd apps/desktop
npx tauri dev
```

> **참고**: `npm run dev`는 브라우저에서 프론트엔드만 확인하는 용도입니다.
> Tauri 네이티브 윈도우로 실행하려면 Rust가 설치된 상태에서 `npx tauri dev`를 사용하세요.

> **Python 3.13 사용 시**: `essentia` 패키지가 아직 Python 3.13을 지원하지 않습니다.
> 오디오 분석 기능이 필요하면 Python 3.11 또는 3.12를 사용하세요.
> 테스트만 실행하려면 `pip install -e ".[dev]"`로 설치하면 됩니다.

---

## 프로젝트 구조

```
ai-driven-development/
├── apps/
│   └── desktop/          # Tauri + React 데스크톱 앱
├── packages/
│   ├── analysis/         # Python: 오디오 분석, ML 추천, 라이브러리 파서
│   ├── backend/          # FastAPI 커뮤니티 백엔드 (인증, 결제, API)
│   └── dsp/              # Rust DSP 엔진, MIDI 출력, 가상 오디오
├── shared/
│   └── types/            # 공유 TypeScript 타입 정의
├── scripts/              # 빌드 및 아키텍처 검증 스크립트
└── docs/
    └── plans/            # 구현 계획 및 아키텍처 문서
```

각 패키지별 상세 문서는 해당 디렉토리의 `README.md`를 참고하세요.

---

## 아키텍처

Clean Architecture 4계층 구조를 따릅니다:

```
Presentation -> Application -> Domain <- Infrastructure
```

| 계층 | 역할 | 의존성 |
|------|------|--------|
| **Domain** | 엔티티, 값 객체, 포트 (인터페이스) | 없음 (순수 비즈니스 로직) |
| **Application** | 유스케이스, 오케스트레이션 | Domain만 |
| **Infrastructure** | DB, 파서, API, 오디오 엔진, ML 런타임 | Domain + Application |
| **Presentation** | UI 컴포넌트, Tauri IPC, 상태 관리 | Application (DTO 통해) |

아키텍처 상세: [docs/plans/PLAN_AI_DJ_ASSIST.md](docs/plans/PLAN_AI_DJ_ASSIST.md)

---

## 주요 기능

- **트랙 분석** - BPM, 키, 에너지, 장르 자동 감지 (Essentia + Chromaprint)
- **AI 추천** - PyTorch MLP 모델 기반 다음 트랙 추천, ONNX로 온디바이스 추론
- **AI 블렌드** - 크로스페이드, EQ 자동화, 필터 스윕, 5가지 블렌드 스타일
- **라이브러리 임포트** - Rekordbox (XML), Traktor (NML), Serato (바이너리) 지원
- **MIDI 출력** - Pioneer, Denon, Native Instruments 장비 연결
- **커뮤니티** - 익명 핑거프린트 공유로 트렌드 확인 (Give-to-Get 모델)
- **분석 대시보드** - 에너지 그래프, Camelot Wheel, 장르 분포 시각화

---

## 분석 CLI (독립 실행)

Tauri 앱 없이 Python 분석 엔진만 단독으로 테스트할 수 있는 CLI 도구입니다.
음악 디렉토리를 지정하면 BPM, 키(Camelot), 에너지를 분석하고, 곡을 선택하면 다음 곡을 추천합니다.

### 설치

```bash
cd packages/analysis
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install librosa          # 또는 essentia (더 정확하지만 설치 까다로움)
```

### 실행

```bash
cd packages/analysis
source .venv/bin/activate

# 전체 라이브러리 분석
python cli.py /path/to/music

# 처음 N곡만 분석 (빠른 테스트)
python cli.py /path/to/music -n 20
```

### CLI 명령어

| 명령 | 단축키 | 설명 |
|------|--------|------|
| `list` | `l` | 분석된 트랙 목록 표시 |
| `select <번호>` | `s <번호>` | 현재 재생 곡 선택 |
| `recommend` | `r` | 선택한 곡 기준 다음 곡 추천 (Top 10) |
| `detail <번호>` | `d <번호>` | 곡 상세 분석 (구간별 에너지 차트 포함) |
| `rescan` | - | 디렉토리 재스캔 및 분석 |
| `help` | `h` | 명령어 도움말 |
| `quit` | `q` | 종료 |

### 사용 예시

```
$ python cli.py /Volumes/Music/basshouse -n 10

============================================================
  AI DJ Assist - Analysis & Recommendation CLI
============================================================

[1/3] Loading analyzer...
  [OK] Librosa fallback loaded
[2/3] Analyzer ready
[3/3] Scanning: /Volumes/Music/basshouse
  Found 1121 audio files (analyzing first 10)

  Analyzing [1/10] Track A... OK  BPM=129.2 Key=9A Energy=91.1
  Analyzing [2/10] Track B... OK  BPM=143.6 Key=2A Energy=91.0
  ...

  [no track] > select 1
  ▶ Now playing: Track A  (BPM=129.2  Key=9A  Energy=91.1)

  [Track A] > recommend

  Now Playing: Track A
  BPM: 129.2  Key: 9A  Energy: 91.1

    #  Score  Title                                BPM     Key Energy  BPM% Key% NRG%
  ───  ─────  ─────────────────────────────────  ──────  ─────── ──────  ──── ──── ────
    1    95%  Track C                              129.2      9A   85.4  100% 100%  80%
    2    93%  Track D                              123.0      9A   88.4   80% 100% 100%
    3    76%  Track E                              129.2      5A   87.0  100%  40% 100%
  ...
```

### 추천 알고리즘

| 요소 | 가중치 | 설명 |
|------|--------|------|
| **Key (Camelot)** | 40% | 같은 키 100%, 인접 키 85%, 상대 조성 90% |
| **BPM** | 35% | 동일 BPM 100%, 더블/하프타임 고려 |
| **Energy** | 25% | 에너지 차이 ±5 이내 100%, 부드러운 전환 우선 |

---

## 테스트

```bash
# 데스크톱 앱 테스트
npm test

# Python 분석 테스트
cd packages/analysis && pytest

# Python 백엔드 테스트
cd packages/backend && pytest

# Rust DSP 테스트
cd packages/dsp && rustup run stable cargo test

# 아키텍처 의존성 검증
npm run check:arch
```

전체 프로젝트에 걸쳐 823개 이상의 테스트가 작성되어 있습니다 (Desktop 325 + Analysis 352 + Backend 146).

---

## 기여하기

기여 방법은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고해 주세요.

## 라이선스

TBD
