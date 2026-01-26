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

### ✅ 완료된 작업
- [x] **프로젝트 초기화**: Vite + Electron + TypeScript + Tailwind 환경 구성 완료
- [x] **Electron 설정**: 투명 윈도우, Always-on-top, IPC 통신 기본 구조 구현
- [x] **데이터베이스**: SQLite 스키마 설계 및 `better-sqlite3` 연동 (Sessions, Tracks, MixLogs 테이블)
- [x] **로그 감시**: `LogWatcher` 서비스 구현 (파일 변경 감지 구조)
- [x] **MIDI 연동**: `MidiListener` 서비스 구현 (DJ 장비 신호 수신 구조)
- [x] **AI 연동**: `AIService` 서비스 구현 (Python 프로세스 실행 및 통신 구조)
- [x] **UI 구현**: React + Tailwind 기반의 플로팅 카드 디자인 및 Zustand 상태 관리 구현

### 🚧 향후 계획 (Next Steps)
- 실제 DJ 소프트웨어(Rekordbox/Serato)의 로그 파일 포맷 분석 및 파싱 로직 구체화
- Python AI 모듈과의 실제 데이터 연동 테스트
- MIDI 매핑 설정 UI 추가 (사용자별 장비 대응)
