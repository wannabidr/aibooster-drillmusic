#DJ Recommender (Raw Audio Only) — 문서화 (Markdown)

목표: “현재 곡(outro)”을 기반으로 다음 곡(intro) 을 자동 추천하는 연속 추천 베이스라인
핵심: CLAP 임베딩 + DJ 관점의 제약(BPM/Key/저역충돌) + 흥(추진력/드럼성/에너지) + 다양성(반복 방지)

⸻

##1) Logic Flow

flowchart TD
  A[build: 오디오 폴더 스캔] --> B[load_audio: soundfile -> 실패 시 librosa]
  B --> C[전역 분석(최대 120s): BPM/Key/에너지/저역/흥/다양성 특성]
  C --> D[intro/outro 분리: intro_seconds/outro_seconds]
  D --> E[mixability: HPSS percussive + RMS 안정성]
  D --> F[CLAP 임베딩: intro_emb, outro_emb]
  F --> G[저장: meta.json + intro/outro embeddings.npy]
  G --> H{FAISS 사용 가능?}
  H -->|Yes| I[faiss_intro.index 생성]
  H -->|No| J[NumPy dot-product 검색 fallback]

  K[recommend: current track 입력] --> L[outro segment 추출 + CLAP 임베딩(q_emb)]
  L --> M[후보 검색: q_emb vs intro_emb topK]
  M --> N[Hard Filter: history 제외 + BPM 호환(half/double 포함)]
  N --> O[Key 필터(선택): tonal clarity 충족 시 호환만 통과]
  O --> P[Re-rank: embed + energy goal + low-end + mix + key + hype + diversity]
  P --> Q[top_k 결과 JSON 출력]

Build 단계 (Indexing)
	•	파일 탐색 → 각 트랙에 대해:
	•	전역 특징(최대 120초): BPM, Key(clarity), energy, low-end, onset_density, brightness, vocal_presence, global percussive_ratio
	•	intro/outro: mixability 점수 + CLAP 임베딩
	•	meta.json, intro_embeddings.npy, outro_embeddings.npy, (옵션) faiss_intro.index 저장

Recommend 단계
	•	현재 곡(outro)의 CLAP 임베딩을 query로 사용
	•	후보 topK 검색 후,
	•	Hard constraints: history 제외, BPM 호환(half/double), (조건부) Key 호환
	•	Soft re-ranking: embed 유사도 + 목표 에너지 곡선 + 저역 충돌 완화 + intro mixability + hype + diversity
	•	최종 top_k 반환

⸻

2) Logic Cons / Adv

✅ Advantages (장점)

DJ 친화적인 “실전 제약”이 있음
	•	**BPM 호환(±tol + half/double time)**로 실제 믹스 가능성↑
	•	**Key 호환(clarity 기반 enforcement)**으로 “키가 애매한 트랙”에서 강제 탈락을 줄임
	•	저역 밴드 충돌 완화(sub/bass/lowmid) → 킥/베이스 겹침 리스크↓

Raw Audio Only로도 “질감” 기반 매칭 가능
	•	메타데이터 없이도 CLAP으로 전환감(다음 트랙 인트로의 성격)을 잡음

흥(그루브/텐션) & 다양성 반영
	•	hype: energy + onset_density + percussive_ratio로 추진력/드럼성 반영
	•	diversity: 최근 히스토리 대비
	•	임베딩 반복 방지(최대 유사도 패널티)
	•	보컬/밝기/저역 성향 “완만한 변주” 유도

실사용 안정성 고려 (Windows/MP3)
	•	soundfile 실패 → librosa.load fallback
	•	resampy 없이 scipy.resample_poly로 리샘플 안정화

⸻

Constraints / Risks (단점/리스크)

CLAP만으로 “구조적 믹스 포인트”를 완전히 보장하진 못함
	•	outro→intro 전환이 실제로 매끄러운지(브레이크다운/필인/무음 등)는 임베딩만으로 한계

Key 추정은 장르/사운드 디자인에 따라 흔들릴 수 있음
	•	특히 베이스하우스/테크노처럼 톤이 약한 트랙은 chroma 기반 key가 불안정할 수 있음
	•	그래서 tonal_clarity threshold가 있지만, 여전히 오탐 가능

저역 충돌 모델이 “정적 비율”이라 실제 마스킹을 완전 반영하진 못함
	•	sidechain, 킥 길이, 베이스 envelope 같은 “시간적 요소”는 반영 X

⸻

3) Future Works

1) 전환(transition) 품질을 직접 모델링
	•	outro→intro 크로스페이드 시뮬레이션 후
	•	저역 충돌(시간축) / RMS 펌핑 / transient 겹침 등을 스코어로 반영
	•	“드롭/브레이크 위치” 탐지 후 믹스 포인트 추천

2) 키/하모닉 강화
	•	Camelot wheel 기반 스코어링 옵션
	•	key 추정 개선:
	•	HPCP 안정화, 장르별 파라미터, 구간별 key-consensus

3) Hype/에너지 커브의 세션 최적화
	•	지금은 곡 단위 “goal up/down/peak”
	•	미래에는 세션 전체를 계획(예: 30분 빌드업 → 피크 → 릴리즈)

4) 학습 가능한 re-ranker 도입
	•	DJ/유저 피드백(스킵/좋아요/재생시간)을 라벨로:
	•	pairwise ranking / bandit 기반 가중치 적응

5) 대규모 인덱싱 최적화
	•	멀티프로세싱(특징추출/임베딩 병렬)
	•	캐시/증분 업데이트(변경된 파일만 재분석)

⸻

4) How to Run / Use

설치

pip install numpy tqdm soundfile librosa scipy torch transformers

⸻

1) 인덱스 생성 (build)

python main.py build \
  --audio_dir ../../Music/basshouse \
  --index_dir ./index \
  --intro_seconds 30 \
  --outro_seconds 30 \
  --bpm_tolerance 4

생성물
	•	index/meta.json
	•	index/intro_embeddings.npy
	•	index/outro_embeddings.npy
	•	(옵션) index/faiss_intro.index

⸻

2) 추천 (recommend)

python main.py recommend \
  --index_dir ./index \
  --current path/to/current.mp3 \
  --goal up \
  --top_k 10 \
  --candidate_k 200

goal 옵션
	•	maintain : 비슷한 에너지 유지
	•	up : 다음 곡이 조금 더 올라가도록
	•	down : 다음 곡이 조금 내려가도록
	•	peak : 강하게 피크로 끌어올림

⸻

3) 연속 세션(코드에서 SessionRecommender)

sr = SessionRecommender(index_dir="./index")
recs = sr.next(current_path="now.mp3", goal="up", top_k=10)

# 선택한 곡을 히스토리에 추가해서 반복 방지
sr.add_played(recs[0]["track_id"])


⸻

5) SDD (Software Design Document)

5.1 목적 (Purpose)
	•	Raw audio만으로 DJ 관점의 “다음 곡 자동 추천(연속 추천)”을 수행한다.
	•	전환감(질감/무드)은 임베딩으로, 실전 믹싱 제약은 룰/특징으로 보강한다.

⸻

5.2 범위 (Scope)

포함
	•	인덱스 빌드(폴더 단위 분석 + 임베딩 저장)
	•	추천(outro→intro 유사도 검색 + 필터 + 재랭킹)
	•	히스토리 기반 다양성 관리

제외(현재)
	•	실제 믹스 오디오 생성/크로스페이드 렌더링
	•	BPM/Key 정답 보장, beatgrid align
	•	온라인 스트리밍/메타데이터 활용

⸻

5.3 아키텍처 개요 (High-level Architecture)

CLI
 ├─ build
 │   ├─ audio loading (soundfile -> librosa)
 │   ├─ feature extraction (BPM/Key/Energy/LowEnd/Hype/Diversity proxies)
 │   ├─ segment intro/outro
 │   ├─ CLAP embedding (48kHz)
 │   └─ persist index (meta + embeddings + optional faiss)
 │
 └─ recommend
     ├─ analyze current track (global features + outro embedding)
     ├─ candidate search (faiss or numpy)
     ├─ hard filters (history/BPM/key)
     └─ rerank (weighted sum) -> top_k


⸻

5.4 주요 컴포넌트 (Components)

1) Audio I/O
	•	load_audio(): soundfile 우선, 실패 시 librosa.load
	•	resample_audio(): scipy resample_poly 기반

2) Feature Extractor
	•	BPM: estimate_bpm()
	•	Key + clarity: estimate_key()
	•	Energy: estimate_energy()
	•	Low-end profile: low_end_profile()
	•	Mixability: mixability_score() (HPSS + RMS 안정성)
	•	Hype proxies:
	•	estimate_onset_density()
	•	estimate_percussive_ratio_global()
	•	Diversity proxies:
	•	estimate_brightness()
	•	estimate_vocal_presence()

3) Embedding
	•	ClapEmbedder: HF CLAP 모델로 intro/outro 임베딩 추출 (48kHz 통일, L2 normalize)

4) Index & Retrieval
	•	save_index()/load_index()
	•	검색:
	•	FAISS 있으면 IndexFlatIP
	•	없으면 NumPy inner product

5) Re-ranking (핵심 의사결정)
	•	총점 =
embed + energy(goal) + low_end + mix_intro + key + hype + diversity
	•	다양성은 최근 8개 히스토리를 기반으로 “너무 비슷한 반복”을 눌러줌

⸻

5.5 데이터 설계 (Data Model)

TrackInfo
	•	정적 메타: track_id, path, duration_s
	•	DJ 제약/성향: bpm, key_root, key_mode, tonal_clarity
	•	에너지/저역: energy_raw, sub, bass, lowmid
	•	믹스: intro_mix, outro_mix
	•	흥/다양성: onset_density, brightness, vocal_presence, perc_ratio_global

IndexMeta
	•	모델/윈도우 설정 + z-score 정규화 통계(energy/onset/brightness/vocal/perc)
	•	전체 tracks: List[TrackInfo]

⸻

5.6 의존성 (Dependencies)
	•	필수: numpy, tqdm, soundfile, librosa, scipy, torch, transformers
	•	선택: faiss (검색 가속)

⸻

5.7 실패/예외 처리 전략 (Reliability)
	•	로딩: soundfile 실패 시 librosa fallback
	•	리샘플: resample_poly 기본 + 무음 방지 가드
	•	build 중 개별 트랙 실패는 누적 후 제외(전체 실패 방지)
	•	DJREC_DEBUG=1이면 traceback 출력

⸻

5.8 성능 고려사항 (Performance)
	•	병목: CLAP 임베딩 추출(트랙당 2회) + librosa 분석
	•	개선 여지:
	•	병렬화, 캐시, FAISS 강제 사용, 임베딩 배치화(가능 시)

⸻
