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
- (개발 예정)

### Cloud
- (개발 예정)

## 프로젝트 구조

```
aibooster-drillmusic/
├── ai/              # AI 추천 엔진 모듈
│   └── src/         # 음악 분석 및 추천 로직
├── app/             # DJ 통합 애플리케이션 (예정)
├── cloud/           # 클라우드 서비스 (예정)
└── README.md        # 프로젝트 소개 문서
```

## 라이선스

(라이선스 정보 추가 예정)

## 참고 자료

- [AI 모듈 상세 문서](ai/src/ReadMe.md)