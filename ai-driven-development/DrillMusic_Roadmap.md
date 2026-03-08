# DrillMusic (AI DJ Assist) — Strategic Roadmap

**Prepared by:** PM Office → CEO
**Date:** March 4, 2026
**Status:** Pre-Development

---

## Executive Summary

DrillMusic is a 12-month build targeting a gap no one owns yet: **AI-powered track recommendation + transition preview for professional DJs**. The roadmap is structured into 4 phases, each ending with a clear milestone that unlocks the next stage of investment and risk.

The core bet: DJs will pay $14.99/month for an AI that knows their library better than they do and can preview blended transitions in under 3 seconds.

---

## High-Level Timeline

```
         Q2 2026              Q3 2026              Q4 2026              Q1 2027
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │   PHASE 1        │  │   PHASE 2        │  │   PHASE 3        │  │   PHASE 4        │
    │   Foundation     │  │   Intelligence   │  │   Community &    │  │   Expansion      │
    │                  │  │                  │  │   Public Launch  │  │                  │
    │   Weeks 1-14     │  │   Weeks 15-26    │  │   Weeks 27-38    │  │   Weeks 39-52    │
    │                  │  │                  │  │                  │  │                  │
    │  → Internal      │  │  → Closed Beta   │  │  → v1.0 Launch   │  │  → v1.5 Windows  │
    │    Alpha         │  │    (50 DJs)      │  │    (Public)      │  │    + Streaming   │
    └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## Phase 1: Foundation (Weeks 1–14) — "Make It Work"

**Goal:** Prove the core loop — analyze a library, recommend tracks, preview a crossfade.

### Deliverables

| Stream | What We Build | Why It Matters |
|--------|--------------|----------------|
| Audio Analysis Engine | BPM, key, energy, fingerprinting (Essentia/librosa) | This is the data foundation everything else depends on |
| Library Import | Rekordbox XML, Serato .crate, Traktor NML | Day-one compatibility with all 3 major DJ platforms |
| Track ID | Chromaprint + AcoustID integration | Enables cross-referencing with community data later |
| Basic Recommendations | Rule-based scoring (BPM + key + energy) | Validates core UX before investing in ML |
| Desktop Shell | Tauri + React + TypeScript | Lightweight, native-feeling app on macOS |
| Beat-Aligned Crossfade | Simple volume fade synced to downbeats | First version of preview — "good enough to test" |

### Milestone: Internal Alpha
- Team can import a 5,000-track Rekordbox library in < 40 min
- Recommendations appear in < 200ms after selecting a track
- Beat-aligned crossfade renders in < 3 sec

### Team Needed
- 1 Full-Stack Engineer (Tauri/React)
- 1 Audio/DSP Engineer (Python + Rust)
- 1 PM (myself)
- Part-time: UI/UX Designer

### Key Risks
- Audio analysis speed on large libraries → Mitigation: prioritize recent/favorited tracks first
- Tauri maturity → Mitigation: Electron as fallback (heavier but proven)

### CEO Decision Point
> At Week 14: Do we see enough signal to invest in ML? If basic rule-based recommendations already feel useful, Phase 2 is a go.

---

## Phase 2: Intelligence (Weeks 15–26) — "Make It Smart"

**Goal:** Replace rules with ML. Build the AI blend engine that becomes our primary upgrade trigger.

### Deliverables

| Stream | What We Build | Why It Matters |
|--------|--------------|----------------|
| ML Recommendation Model | PyTorch model trained on seed data (1001Tracklists) | Step change in recommendation quality |
| Genre Classification | Spectral/rhythmic feature embeddings | Enables genre-aware suggestions |
| AI Blend Engine | EQ automation, filter sweeps, phrase detection (Rust DSP) | **This is the feature people pay for** |
| Mix History Import | Parse personal history from Rekordbox/Serato/Traktor | Personalizes recommendations from day one |
| Camelot Wheel + Energy Graph | Visual UI for harmonic mixing and energy arcs | DJs think in Camelot — we speak their language |

### Milestone: Closed Beta (50 DJs)
- Recruit 50 professional DJs (clubs/festivals)
- Recommendation acceptance rate > 30% (target 40% at launch)
- AI blend quality rated "usable" by > 60% of beta testers
- NPS > 30 among beta cohort

### Team Growth
- Add: 1 ML Engineer
- Add: 1 Full-time UI/UX Designer
- Total: 5 people

### Key Risks
- AI blend quality may not meet DJ standards → Mitigation: beat-aligned crossfade always available as fallback; genre-specific fine-tuning
- Training data quality from public sources → Mitigation: manual curation of seed dataset; beta DJ feedback loop

### CEO Decision Point
> At Week 26: Beta metrics review. If recommendation acceptance > 30% and blend quality feedback is positive, we greenlight public launch + Stripe integration. This is our go/no-go for commercialization.

---

## Phase 3: Community & Public Launch (Weeks 27–38) — "Make It a Product"

**Goal:** Ship v1.0. Turn the tool into a business.

### Deliverables

| Stream | What We Build | Why It Matters |
|--------|--------------|----------------|
| Community Backend | FastAPI + PostgreSQL on AWS/GCP | Enables collaborative filtering ("DJs who played A → B") |
| Give-to-Get Model | Share history → unlock community insights | Creates data flywheel without forcing participation |
| Mixcloud/SoundCloud Import | Parse public tracklists | More data sources = better cold-start experience |
| Dark Theme (Club-Optimized) | High-contrast, glanceable UI | Usable at 3AM in a dark booth |
| Stripe Subscriptions | Free / Pro Monthly ($14.99) / Pro Annual ($119.99) | Revenue! |
| User Preference Tuning | Adjust recommendation weights per user | Personalization differentiator |

### Milestone: Public Launch (v1.0)
- Available on our website for macOS (Apple Silicon + Intel)
- Free tier fully functional (no credit card required)
- Stripe billing live
- 500+ registered users in first 30 days (target)
- 5% free-to-paid conversion in first 60 days

### Team (same size, shift focus)
- Full-Stack → billing, backend, community features
- Audio/DSP → blend quality iteration based on beta feedback
- ML → collaborative filtering model
- UI/UX → onboarding, dark theme polish

### Key Risks
- Low community opt-in → Mitigation: seed dataset from 1001Tracklists ensures recommendations work without community data
- Free tier too good → Mitigation: AI blend (the "wow" feature) is Pro-only
- Competing announcement from Pioneer/Serato → Mitigation: ship fast, cross-platform advantage

### CEO Decision Point
> At Week 38 (Launch + 2 weeks): Review conversion rate, retention, and NPS. If unit economics look viable, authorize Phase 4 hiring for Windows + streaming.

---

## Phase 4: Expansion (Weeks 39–52+) — "Make It Big"

**Goal:** Expand TAM. Windows doubles our addressable market. Streaming integration removes library friction.

### Deliverables

| Stream | What We Build | Why It Matters |
|--------|--------------|----------------|
| Windows App | Tauri cross-compilation | ~40% of DJs use Windows |
| Streaming Integration | Beatport Link, Tidal, Spotify catalog browsing | Recommend from millions of tracks, not just local library |
| MIDI Output | Send BPM/key data to hardware | Deeper integration with DJ workflows |
| Virtual Audio Cable | Route preview audio to headphones | Audition blends without disrupting main output |
| Advanced Analytics | Set energy analysis, genre distribution, mixing patterns | Power-user retention feature |
| Label/Promoter API | External access to anonymized trend data | B2B revenue stream |

### Milestone: v1.5 Launch
- Windows parity with macOS
- At least 1 streaming integration live
- Monthly recurring revenue trajectory toward break-even

### Team Growth
- Add: 1 Windows/Cross-platform Engineer
- Add: 1 Backend Engineer (API, streaming integrations)
- Total: 7 people

---

## Resource Summary

| Phase | Duration | Team Size | Key Hire |
|-------|----------|-----------|----------|
| Phase 1 | 14 weeks | 3.5 | Audio/DSP Engineer |
| Phase 2 | 12 weeks | 5 | ML Engineer, UI/UX Designer |
| Phase 3 | 12 weeks | 5 | (no new hires) |
| Phase 4 | 14+ weeks | 7 | Windows Engineer, Backend Engineer |

**Total estimated headcount to v1.0 launch: 5 people over ~9 months.**

---

## Revenue Projection (Conservative)

| Month Post-Launch | Registered Users | Paid Subscribers | MRR |
|-------------------|-----------------|------------------|-----|
| Month 1 | 500 | 25 (5%) | $375 |
| Month 3 | 2,000 | 150 (7.5%) | $2,250 |
| Month 6 | 5,000 | 500 (10%) | $7,500 |
| Month 12 | 15,000 | 2,000 (13%) | $30,000 |

*Assumes blended ARPU of ~$15 (mix of monthly and annual). Annual subscribers modeled at $10/mo effective.*

---

## What I Need From You, CEO

1. **Hiring approval** — Audio/DSP Engineer is the critical first hire. Without this person, nothing starts. Ideally onboard by Week 1.
2. **Budget sign-off** — Cloud infrastructure (AWS/GCP) + Stripe fees + 1001Tracklists data acquisition. Estimated ~$2,000/month pre-launch.
3. **Go/No-Go cadence** — I've built 3 decision gates into the roadmap (Weeks 14, 26, 38). Each one is a checkpoint where we either accelerate or pivot.
4. **Beta DJ network** — Do you have connections in the Korean or international DJ scene? We need 50 professional DJs for closed beta by Week 15.

---

*This roadmap is a living document. I'll update it bi-weekly as we learn more.*
