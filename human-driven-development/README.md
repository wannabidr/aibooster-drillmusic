# PNU AI Booster - DrillMusic (AI DJ Project)

## 프로젝트 소개

DrillMusic은 **AI DJ 시스템**을 구현하는 프로젝트입니다. PNU AI Booster 프로그램의 일환으로, AI Tool을 활용한 AI Native 개발 방식을 통해 전문 DJ의 워크플로우를 혁신하는 것을 목표로 합니다.

PNU AI Booster는 AI Tool을 적극 활용하여 프로젝트를 개발하면서 AI Native 개발자를 양성하는 교육 프로그램입니다. DrillMusic은 이러한 철학을 바탕으로 AI, App, Cloud 세 분야의 개발자가 협업하여 실용적인 AI DJ 솔루션을 구현합니다.

## 팀 구성

**Team AISFA**

- **김성욱: Team Leader, PM, Application Developer**: DJ 워크플로우 통합 애플리케이션 개발
- **문경환: AI Developer**: 음악 분석 및 추천 알고리즘 개발
- **전진혁: Cloud Engineer**: 데이터 저장 및 관리 인프라 구축

## 프로젝트 목표

### 🎵 AI - 음악 분석 및 추천 엔진

DJ 믹싱이 자연스럽게 이어지도록 하는 AI 로직을 개발합니다.

- **음악 분석**: 오디오 신호에서 BPM, Key, 에너지, 저역 프로파일 등의 특징(feature) 추출
- **스마트 추천**: DJ의 의도(에너지 상승/유지/하락)에 맞춘 다음 곡 자동 추천
  - CLAP 임베딩 기반 음질감 매칭
  - BPM 호환성, Key 호환성 등 실전 믹싱 제약 반영
  - 히스토리 기반 다양성 관리로 반복 방지
- **Mixset 데이터 활용**: 인터넷의 mixset 데이터를 활용하여 검증된 트랜지션 패턴 학습 및 적용

### 📱 App - DJ 통합 애플리케이션

전문 DJ 프로그램(예: Rekordbox, Serato, Traktor)과 함께 사용할 수 있는 위젯 형식의 애플리케이션을 개발합니다.

- **위젯 인터페이스**: DJ 프로그램과 동시에 실행 가능한 참고용 위젯
- **실시간 연동**: 현재 재생 중인 곡 정보를 자동으로 인식
- **AI 추천 실행**: 현재 곡을 기반으로 AI 로직을 실행하여 다음 곡 추천
- **직관적 UI**: 믹싱 중 빠르게 확인할 수 있는 간결하고 명확한 인터페이스

### ☁️ Cloud - 데이터 저장 및 관리

DJ의 재생 기록과 음악 라이브러리 정보를 안정적으로 저장하고 관리합니다.

- **재생 기록 저장**: 재생한 곡의 정보, 재생 시간, 순서 등 히스토리 관리
- **음악 메타데이터**: 분석된 특징(feature) 및 인덱스 데이터 저장
- **세션 관리**: DJ 세션별 재생 목록 및 통계 관리
- **확장성**: 대규모 음악 라이브러리 지원 및 빠른 검색을 위한 인덱싱

## 기술 스택

### AI
- Python, PyTorch
- CLAP (Contrastive Language-Audio Pre-training) 모델
- librosa, soundfile (오디오 처리)
- FAISS (고속 유사도 검색)
- NumPy, SciPy (수치 계산)

### App
- Electron (Desktop App Framework)
- React + TypeScript + Vite
- Tailwind CSS (스타일링)
- Zustand (상태 관리)
- better-sqlite3 (로컬 데이터베이스)
- node-midi (MIDI 장비 연동)
- chokidar (파일 감시)

### Cloud
- (개발 예정)

## 프로젝트 구조

```
aibooster-drillmusic/
├── ai/              # AI 추천 엔진 모듈
│   └── src/         # 음악 분석 및 추천 로직
│       ├── main.py  # CLI 및 서버 모드
│       └── index/   # FAISS 인덱스 저장소
├── app/             # DJ 통합 애플리케이션 ✅ 완료
│   ├── electron/    # Electron 메인 프로세스
│   │   ├── main.ts
│   │   └── services/
│   ├── src/         # React 프론트엔드
│   └── __tests__/   # 테스트 (31개, 100% 통과)
├── cloud/           # 클라우드 서비스 (예정)
└── README.md        # 프로젝트 소개 문서
```

## 🚀 프로젝트 진행 상황 (2026-01-26 기준)

### ✅ AI 모듈
- [x] CLAP 임베딩 기반 음악 추천 엔진
- [x] BPM/Key 호환성 필터링
- [x] 에너지 목표(maintain/up/down/peak) 지원
- [x] FAISS 기반 고속 검색
- [x] stdin/stdout JSON 통신 서버 모드

### ✅ App 모듈
- [x] Electron 기반 플로팅 위젯 (300x150, Always-on-top)
- [x] 로그 파일 실시간 감시 (chokidar)
- [x] MIDI 컨트롤러 연동 (node-midi)
- [x] Python AI 모듈과의 IPC 통신
- [x] SQLite 데이터베이스 (세션, 트랙, 믹싱 로그)
- [x] React UI (실시간 추천 표시)
- [x] TDD 테스트 (31개 테스트, 100% 통과)

### 🚧 Cloud 모듈
- [ ] 개발 예정

---

## 🖥️ 시스템 요구사항

- **OS**: macOS 10.15 (Catalina) 이상
- **Node.js**: v18 이상
- **Python**: 3.8 이상
- **uv**: Python 패키지 관리자 ([설치 방법](https://github.com/astral-sh/uv))
- **공간**: 최소 500MB (AI 모델 및 인덱스 포함)

---

## 🚀 설치 및 실행 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd aibooster-drillmusic
```

### 2. Python AI 모듈 설정

#### uv 설치 (처음 한 번만)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 Homebrew
brew install uv

# 설치 확인
uv --version
```

#### 가상환경 생성 및 의존성 설치

프로젝트 루트에서 통합 가상환경을 생성하고 의존성을 설치합니다.

```bash
# 프로젝트 루트 디렉토리 (aibooster-drillmusic)

# 설치 확인
python --version

# 가상환경 활성화
source .venv/bin/activate  # macOS/Linux
# 또는
.venv\Scripts\activate     # Windows

# 가상환경 생성 및 의존성 설치 (pyproject.toml 기반)
uv sync
```

#### AI 인덱스 빌드

```bash
# 가상환경이 활성화된 상태에서 실행
python ai/src/main.py build \
  --audio_dir ~/Music/DJ \
  --index_dir ./ai/src/index \
  --intro_seconds 30 \
  --outro_seconds 30 \
  --bpm_tolerance 4

# 빌드 완료 확인
ls -la ./ai/src/index/
# 예상 출력: faiss_intro.index, intro_embeddings.npy, outro_embeddings.npy, meta.json
```

**참고**: 
- 첫 빌드는 음악 파일 수에 따라 시간이 소요됩니다 (100곡당 약 5-10분)
- uv는 pip보다 10-100배 빠른 의존성 설치 속도를 제공합니다
- 가상환경은 프로젝트별로 독립적인 Python 환경을 제공합니다

### 3. Electron 앱 설정 및 실행

**중요**: Electron 앱은 Python AI 모듈을 실행하므로, 프로젝트 루트의 가상환경이 활성화되어 있어야 합니다.

```bash
# 프로젝트 루트에서 가상환경 활성화
source .venv/bin/activate  # macOS/Linux
# 또는
.venv\Scripts\activate     # Windows

# 앱 디렉토리로 이동
cd app

# Node.js 의존성 설치
npm install

# 개발 모드로 실행 (가상환경이 활성화된 터미널에서)
npm run dev
```

앱이 화면 우측 하단에 플로팅 위젯으로 표시됩니다.

**참고**: 
- Electron 앱은 `python3` 명령어로 AI 모듈을 실행합니다. 가상환경이 활성화되어 있으면 가상환경의 Python이 사용됩니다.
- 별도 터미널에서 실행하는 경우, 프로젝트 루트의 가상환경을 먼저 활성화하거나 시스템 Python에 AI 모듈 의존성이 설치되어 있어야 합니다.

### 4. 테스트 실행 (선택사항)

```bash
# 앱 디렉토리에서
npm test

# 특정 테스트만 실행
npm test -- electron/__tests__/Database.test.ts

# Watch 모드
npm run test:watch
```

### 5. 프로덕션 빌드 (선택사항)

```bash
# 앱 빌드
npm run build

# 배포용 패키지 생성
npm run dist
```

---

## 📖 사용 방법

### 초기 설정

1. **AI 인덱스 준비**: 위의 2단계에서 음악 라이브러리 인덱스를 생성합니다.

2. **로그 파일 경로 설정**: DJ 소프트웨어의 로그 파일 경로를 지정합니다.
   ```bash
   # 테스트용 더미 로그 파일 생성
   echo "$(date '+%Y-%m-%d %H:%M:%S') | Skrillex - Scary Monsters" >> ~/drillmusic_test.log
   ```
   
   앱의 `electron/main.ts`에서 로그 파일 경로를 설정할 수 있습니다.

3. **MIDI 장비 연결 (선택사항)**: DJ 컨트롤러를 USB로 연결하면 자동으로 감지됩니다.

### 기본 사용 흐름

1. **DJ 소프트웨어에서 곡 재생** → LogWatcher가 감지
2. **자동 곡 인식** → "NOW PLAYING"에 곡 정보 표시
3. **AI 추천 생성** → "NEXT UP"에 Top 3 추천 표시
4. **추천 곡 활용** → BPM과 함께 표시되어 믹싱 호환성 확인

### 위젯 화면

```
┌─────────────────────────────┐
│ 🎵 DrillMusic AI      🟢    │  ← 헤더
├─────────────────────────────┤
│ NOW PLAYING                 │
│ Track Title                 │  ← 현재 곡
│ Artist Name                 │
│ 128 BPM • Am                │
├─────────────────────────────┤
│ 🎵 NEXT UP                  │
│ 1. Recommended Track 1  128 │  ← AI 추천
│ 2. Recommended Track 2  130 │
│ 3. Recommended Track 3  126 │
└─────────────────────────────┘
```

---

## 🐛 문제 해결

### AI 추천이 표시되지 않을 때

```bash
# AI 인덱스 확인
ls -la ai/src/index/
# meta.json, faiss_intro.index 등이 있어야 함

# 인덱스 재빌드 (프로젝트 루트에서)
source .venv/bin/activate  # 가상환경 활성화
python ai/src/main.py build --audio_dir ~/Music/DJ --index_dir ./ai/src/index
```

### 가상환경 관련 문제

```bash
# 가상환경이 없을 때 (프로젝트 루트에서)
uv sync
source .venv/bin/activate

# 의존성 업데이트
uv sync
```

### 로그 파일이 감지되지 않을 때

```bash
# 테스트 로그 파일로 테스트
echo "$(date '+%Y-%m-%d %H:%M:%S') | Test Artist - Test Track" >> ~/drillmusic_test.log
```

### 앱이 시작되지 않을 때

```bash
# 로그 확인
cd app
npm run dev

# 데이터베이스 초기화
rm ~/Library/Application\ Support/app/drillmusic.db
```

---

## 📚 참고 문서

- **AI 모듈**: [ai/src/ReadMe.md](ai/src/ReadMe.md) - AI 추천 알고리즘 상세 설명
- **App 모듈**: [app/README.md](app/README.md) - 앱 기술 문서 및 개발자 가이드
- **테스트 결과**: [app/TEST_RESULTS.md](app/TEST_RESULTS.md) - TDD 테스트 결과 보고서

---

## 🤝 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Run tests (`cd app && npm test`)
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

---

## 📝 라이선스

This project is licensed under the MIT License.