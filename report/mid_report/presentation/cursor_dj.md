# Dj 통합 어플리케이션 기획
_Exported on 2026. 1. 28. at 17시 37분 49초 GMT+9 from Cursor (2.3.41)_

---

**User**

@README.md 의 App - DJ 통합 어플리케이션을 참고해서 app을 제작할 것이다. 기획사항을 먼저 완성하고자 한다. 추천하는 기능이나 프로젝트 구성에 대해 알려줘. spec-driven-development에 따라서 작성할 예정이니 필요한 것에 대해서 꼼꼼하게 물어봐라.

---

**Cursor**

README.md의 **App - DJ 통합 애플리케이션** 섹션과 프로젝트 구조를 바탕으로 기획 및 프로젝트 구성을 제안해 드립니다. Spec-driven development(사양 주도 개발) 방식에 따라, 개발 전에 명확한 요구사항을 정의하기 위해 몇 가지 중요한 질문을 드리겠습니다.

우선, 제가 제안하는 **기본 기획 및 프로젝트 구성안**입니다.

### 📱 DrillMusic App 기획 제안

**1. 핵심 컨셉: "Unintrusive Assistant (방해받지 않는 조수)"**
DJ가 메인으로 사용하는 프로그램(Rekordbox, Serato 등)을 가리지 않으면서, 필요할 때 곁눈질로 정보를 확인할 수 있는 **컴팩트한 위젯 형태**를 지향합니다.

**2. 주요 기능 (Feature Spec)**
*   **Always-on-top Widget**: 다른 창보다 항상 위에 떠 있는 미니 모드 지원.
*   **Now Playing Detection (현재 곡 인식)**: DJ 소프트웨어에서 재생 중인 곡을 실시간 감지.
*   **Smart Recommendation View**:
    *   현재 곡과 어울리는 다음 곡 3~5개 추천.
    *   추천 이유 표시 (예: "에너지 유지", "BPM 호환", "조화로운 Key").
    *   각 추천 곡의 매칭 점수 시각화.
*   **Library Sync**: 로컬 음악 파일 또는 Rekordbox XML과 메타데이터 동기화.

**3. 기술 스택 제안**
*   **Framework**: **Electron** (React + TypeScript)
    *   *이유*: 위젯 형태의 투명 창, 항상 위에 보기(Always on top), 시스템 레벨 접근(파일 읽기 등) 구현이 용이하며, 빠른 UI 개발이 가능합니다.
*   **UI Library**: Tailwind CSS + Shadcn/ui (빠르고 현대적인 디자인)
*   **State Management**: Zustand (가볍고 직관적)

---

### ❓ 상세 Spec 확정을 위한 질문 (Critical Questions)

본격적인 개발에 앞서, 다음 사항들에 대해 답변해 주시면 구체적인 **기술 명세서(Technical Spec)**를 작성하겠습니다.

**Q1. 타겟 플랫폼 (OS)**
현재 개발 환경은 macOS이지만, 최종 배포 대상은 어디까지 고려하시나요?
*   **A**: macOS 전용 (개발 용이성 우선)
*   **B**: Windows + macOS 크로스 플랫폼 (대부분의 클럽 환경이 Windows일 수 있음)
    *   *(추천: 일단 macOS로 시작하되, 크로스 플랫폼이 가능한 Electron을 사용)*

**Q2. "현재 재생 중인 곡" 인식 방식 (가장 중요한 기술적 과제)**
어떤 방식으로 현재 곡을 알아낼지 결정해야 합니다.
*   **A: 파일/로그 트래킹 (권장)**: Rekordbox/Serato가 남기는 `History` 파일이나 `XML` 파일을 실시간으로 감시(Watch)하여 변경 사항을 읽습니다. 가장 안정적입니다.
*   **B: 오디오 핑거프린팅**: 시스템 오디오를 캡처하여 AI 모델로 무슨 곡인지 식별합니다. (정확하지만 리소스 소모 큼)
*   **C: 화면 캡처 (OCR)**: DJ 프로그램 화면의 특정 영역 텍스트를 인식합니다. (불안정할 수 있음)

**Q3. AI 엔진 연동 방식**
`ai/` 폴더의 Python 로직을 어떻게 실행할까요?
*   **A: 로컬 프로세스 실행 (Child Process)**: App이 `python main.py`를 서브 프로세스로 직접 실행하고, 표준 입출력(stdio)으로 통신합니다. (간단함)
*   **B: 로컬 API 서버**: AI 모듈을 FastAPI/Flask 서버로 띄우고, App이 `localhost:8000`으로 요청을 보냅니다. (구조적으로 깔끔하고 분리됨)

**Q4. 디자인/UI 레이아웃 선호도**
*   **A: 가로형 바 (Ticker)**: 화면 상단이나 하단에 얇게 위치.
*   **B: 플로팅 카드**: 스마트폰 화면처럼 우측 구석에 떠 있는 형태.

답변을 주시면 이를 바탕으로 **프로젝트 초기 세팅(Scaffolding)** 및 **상세 할 일 목록(To-do List)**을 작성하겠습니다.

---

**User**

Q1. 일단 MacOS 전용으로 개발해라.
Q2. 파일/로그 트래킹으로 한다. rekordbox/serato에서 현재 재생 중인 곡을 가져와서, 그것을 우리 앱의 데이터베이스에 따로 저장해야한다. 곡 정보 뿐만 아니라 디제이가 어떻게 믹싱하는지(곡의 몇 초부터 재생하는지, 어떤 이펙트를 사용하는지, 페이더는 얼마나 올리고 내리는지, 이퀄라이저는 어떻게 조정하는지 등) 정보도 같이 저장했으면 좋겠다.
Q3. 일단 A 로컬 프로세스 실행으로 한다. 하지만 추후 B로 할 수 있음을 참고해라
Q4. 플로팅 카드 형태로 제작한다.

상세 코드는 작성하지말고 초기 세팅과 상세 할 일 목록만 작성해라.

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

파일을 처음 만들 때, command를 통해 파일을 먼저 생성하고 거기다가 내용 작성해라.

마저 진행하자

파일 touch로 생성하고 내용 작성하라고

vite-env.d.ts부터 마저 내용 작성

테스트는 작성하지 말고 기획내용을 따로 어디 적어놨으면 좋겠는데. 기획내용과 프로젝트 진행 상황을 @app/README.md 에 한국어로 작성해라. 

