# Contributing to AI DJ Assist

## Workflow

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Run tests and linting
5. Open a pull request

## Branch Naming

- `feature/short-description` -- new features
- `fix/short-description` -- bug fixes
- `docs/short-description` -- documentation only

## Before Submitting a PR

Run the full test suite and architecture check:

```bash
# Desktop (TypeScript)
npm test
npm run lint
npm run type-check

# Analysis (Python)
cd packages/analysis && pytest

# Backend (Python)
cd packages/backend && pytest

# DSP (Rust)
cd packages/dsp && rustup run stable cargo test

# Architecture dependency guard
npm run check:arch
```

## Architecture Rules

This project follows **Clean Architecture** with strict layer separation:

```
Presentation -> Application -> Domain <- Infrastructure
```

- **Domain** must have zero external dependencies (stdlib only)
- **Application** depends only on Domain
- **Infrastructure** implements Domain ports (dependency inversion)
- **Presentation** calls Application use cases via DTOs

Run `scripts/check-architecture.sh` to verify no dependency violations.

## Code Style

| Language | Formatter | Linter |
|----------|-----------|--------|
| Python | black | ruff |
| TypeScript | prettier | eslint |
| Rust | cargo fmt | clippy |

Pre-commit hooks (via Husky + lint-staged) will auto-format TypeScript files on commit.
