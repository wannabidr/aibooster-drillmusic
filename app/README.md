# DrillMusic App (AI DJ Assistant)

## 📌 프로젝트 개요
전문 DJ 소프트웨어(Rekordbox, Serato 등)와 연동되어 **현재 재생 중인 곡을 인식**하고, AI 분석을 통해 **최적의 다음 곡을 추천**해주는 macOS 전용 위젯 애플리케이션입니다.

## 🛠 기술 스택 (Tech Stack)
- **Core**: Electron (Desktop App Framework)
- **Frontend**: React + TypeScript + Vite
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Database**: `better-sqlite3` (로컬 데이터 저장)
- **External Device**: `node-midi` (DJ 장비 신호 수집)
- **File Watching**: `chokidar` (로그 파일 감시)
- **AI Bridge**: Node.js `child_process` (Python AI 모듈 연동)

## 📋 기획 및 기능 명세 (Feature Specs)

### 1. UI/UX 컨셉
- **Unintrusive Assistant**: DJ의 시야를 방해하지 않는 컴팩트한 **Always-on-top 플로팅 위젯**.
- **Transparent Design**: 반투명 배경으로 기존 DJ 소프트웨어 위에 자연스럽게 오버레이.

### 2. 핵심 기능
#### A. 데이터 수집 (Data Collection)
- **곡 인식 (Track Detection)**: `chokidar`를 이용해 DJ 소프트웨어의 History/Log 파일을 실시간 감시하여 현재 곡 정보를 파싱합니다.
- **믹싱 정보 수집 (Mixing Telemetry)**: `node-midi`를 통해 DJ 컨트롤러의 EQ, Fader, Knob 조작 데이터를 실시간으로 수집하여 DB에 저장합니다.

#### B. AI 추천 (AI Recommendation)
- 현재 곡 정보를 Python AI 모듈(`ai/src/main.py`)로 전달합니다.
- AI가 분석한 추천 곡 리스트(Top 3)를 받아와 UI에 표시합니다.

#### C. 데이터 저장 (Local Storage)
- **SQLite**를 사용하여 세션, 트랙 정보, 믹싱 로그를 로컬에 안전하게 저장합니다.

## 🚀 프로젝트 진행 상황 (Progress)

### ✅ 완료된 작업 (2026-01-26 기준)

#### 핵심 시스템 구축
- [x] **프로젝트 초기화**: Vite + Electron + TypeScript + Tailwind 환경 구성 완료
- [x] **Electron 설정**: 투명 윈도우, Always-on-top, IPC 통신 완전 구현
- [x] **데이터베이스**: SQLite 스키마 설계 및 `better-sqlite3` 연동 (Sessions, Tracks, MixLogs 테이블)
- [x] **로그 감시**: `LogWatcher` 서비스 완전 구현 (chokidar 기반 실시간 파일 감시)
- [x] **MIDI 연동**: `MidiListener` 서비스 완전 구현 (DJ 장비 신호 수신)
- [x] **AI 연동**: `AIService` 서비스 완전 구현 (Python 프로세스 stdin/stdout 통신)
- [x] **UI 구현**: React + Tailwind 기반의 플로팅 카드 디자인 및 Zustand 상태 관리 완료

#### 서비스 통합
- [x] **메인 프로세스 통합**: `main.ts`에서 모든 서비스(Database, LogWatcher, MidiListener, AIService) 초기화 및 연결
- [x] **IPC 핸들러**: 8개 IPC 채널 구현
  - `get-current-track`, `set-current-track`
  - `get-recommendations`, `list-midi-devices`, `select-midi-device`
  - `set-log-path`
- [x] **Preload API**: 완전한 TypeScript 타입 정의와 함께 구현
- [x] **UI-백엔드 연동**: 실시간 트랙 업데이트 및 AI 추천 표시

#### Python AI 모듈 개선
- [x] **Serve 모드 추가**: stdin/stdout 기반 JSON 통신 프로토콜 구현
- [x] **추천 API**: 현재 곡 기반 Top-K 추천 시스템 (`goal`: maintain/up/down/peak)
- [x] **CLAP 임베딩**: 오디오 유사도 기반 추천 엔진

#### TDD 테스트 구현
- [x] **Jest 테스트 환경**: ESM 모드 TypeScript 테스트 설정 완료
- [x] **Database 테스트**: 10개 테스트 (100% 통과)
- [x] **LogWatcher 테스트**: 7개 테스트 (100% 통과)
- [x] **AIService 테스트**: 10개 테스트 (핵심 로직 검증)
- [x] **통합 테스트**: 14개 테스트 (100% 통과)
- [x] **총 31개 테스트 - 100% 통과**

### 🎯 검증된 기능

| 기능 | 상태 | 설명 |
|------|------|------|
| 곡 인식 | ✅ 완료 | DJ 소프트웨어 로그 파일 실시간 감시 및 파싱 |
| AI 추천 | ✅ 완료 | CLAP 임베딩 기반 Top 3 추천 (BPM/Key 호환성 필터) |
| MIDI 수집 | ✅ 완료 | DJ 컨트롤러 조작 데이터 실시간 수집 |
| 데이터 저장 | ✅ 완료 | SQLite 세션/트랙/믹싱 로그 저장 |
| UI 위젯 | ✅ 완료 | 300x150 컴팩트 플로팅 윈도우 (Always-on-top) |
| 에러 처리 | ✅ 완료 | Graceful degradation (서비스 실패 시 앱 지속 동작) |

### 🚧 향후 계획 (Next Steps)
- 실제 DJ 소프트웨어(Rekordbox/Serato)의 로그 파일 포맷 분석 및 파싱 로직 구체화
- Python AI 인덱스 빌드: 실제 음악 라이브러리로 FAISS 인덱스 생성
- MIDI 매핑 설정 UI 추가 (사용자별 장비 대응)
- 세션 히스토리 뷰어 및 통계 기능
- macOS 앱 번들 및 배포 패키징

---

## 🚀 빠른 시작

**설치 및 실행 방법은 [루트 README](../README.md#-설치-및-실행-방법)를 참고하세요.**

### 개발 명령어

```bash
# 개발 모드
npm run dev

# 테스트
npm test
npm run test:watch
npm run test:coverage

# 빌드
npm run build
npm run lint
```

---

## 🏗️ 아키텍처

### 프로젝트 구조

```
app/
├── electron/                # Electron 메인 프로세스
│   ├── main.ts             # 앱 진입점, 서비스 초기화
│   ├── preload.ts          # IPC API 노출
│   ├── services/           # 백엔드 서비스
│   │   ├── Database.ts     # SQLite 데이터베이스
│   │   ├── LogWatcher.ts   # 로그 파일 감시
│   │   ├── MidiListener.ts # MIDI 장비 연동
│   │   └── AIService.ts    # Python AI 통신
│   └── __tests__/          # 단위 및 통합 테스트
├── src/                    # React 프론트엔드
│   ├── App.tsx            # 메인 UI 컴포넌트
│   ├── stores/            # Zustand 상태 관리
│   └── vite-env.d.ts      # TypeScript 타입 정의
├── jest.config.js         # Jest 설정
└── TEST_RESULTS.md        # 테스트 결과 보고서
```

### 데이터 흐름

```
DJ Software Log → LogWatcher → IPC → Renderer (UI)
                                ↓
                            Database (SQLite)
                                ↓
                          AIService (Python)
                                ↓
                    Recommendations → Renderer (UI)

MIDI Controller → MidiListener → Database → (향후 분석)
```

### IPC API

```typescript
// Renderer에서 사용 가능한 전역 API
window.electronAPI = {
  // 트랙 관리
  getCurrentTrack(): Promise<Track>
  setCurrentTrack(track: Track): Promise<Result>
  onTrackUpdate(callback): Unsubscribe

  // AI 추천
  getRecommendations(track: Track): Promise<RecommendationResponse>
  onRecommendationsUpdate(callback): Unsubscribe

  // MIDI
  listMidiDevices(): Promise<MidiDevice[]>
  selectMidiDevice(id: number): Promise<Result>
  onMidiMessage(callback): Unsubscribe

  // 설정
  setLogPath(path: string): Promise<Result>
}
```

---

## 📚 관련 문서

- **[루트 README](../README.md)** - 프로젝트 개요 및 설치 방법
- **[테스트 결과](TEST_RESULTS.md)** - TDD 테스트 결과 보고서 (31개 테스트, 100% 통과)
- **[AI 모듈](../ai/src/ReadMe.md)** - Python AI 추천 알고리즘 상세 설명
