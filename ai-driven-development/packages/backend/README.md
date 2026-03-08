# Backend Package

FastAPI community backend for AI DJ Assist.

## Features

- **Authentication**: OAuth 2.0 (Google, Apple), RS256 JWT with refresh token rotation
- **Billing**: Stripe integration (Pro Monthly $14.99, Annual $119.99, 14-day trial)
- **Community Sync**: Shared mix history and crowd-sourced data
- **B2B API**: 3-tier API key auth, trend aggregation, rate limiting
- **Privacy**: GDPR-compliant, give-to-get gating, right-to-deletion

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires PostgreSQL running locally or via Docker.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AIDJ_DATABASE_URL` | PostgreSQL connection string |
| `AIDJ_JWT_PRIVATE_KEY_PATH` | RS256 private key PEM file |
| `AIDJ_JWT_PUBLIC_KEY_PATH` | RS256 public key PEM file |
| `AIDJ_GOOGLE_CLIENT_ID` | Google OAuth client ID |
| `AIDJ_GOOGLE_CLIENT_SECRET` | Google OAuth client secret |
| `AIDJ_APPLE_CLIENT_ID` | Apple OAuth client ID |
| `AIDJ_STRIPE_SECRET_KEY` | Stripe API secret key |
| `AIDJ_STRIPE_WEBHOOK_SECRET` | Stripe webhook signing secret |
| `AIDJ_CORS_ORIGINS` | Allowed CORS origins |

See `.env.example` in the project root for a template.

## Test

```bash
pytest
```
