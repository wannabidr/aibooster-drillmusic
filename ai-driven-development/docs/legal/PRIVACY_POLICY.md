# AI DJ Assist - Privacy Policy

**Effective Date**: [INSERT DATE BEFORE LAUNCH]
**Last Updated**: 2026-03-07

## 1. Introduction

AI DJ Assist ("we", "our", "the Service") is a desktop companion application for professional DJs. This Privacy Policy explains how we collect, use, store, and protect your information when you use our application and community features.

## 2. Data We Collect

### 2.1 Account Data

When you create an account via Google or Apple Sign-In, we collect:

- **Email address** (from your OAuth provider)
- **Display name** (optional, from your OAuth provider)
- **OAuth provider identifier** (a unique ID from Google or Apple)

We do NOT collect or store your OAuth provider password.

### 2.2 Local Analysis Data (Stays on Your Device)

The following data is generated and stored **entirely on your device** and is never transmitted to our servers:

- Audio file metadata (BPM, key, energy, genre)
- Audio fingerprints
- Your DJ library structure and playlists
- Recommendation history
- User preferences and settings

### 2.3 Community Data (Opt-In Only)

If you choose to participate in the community feature ("Give-to-Get"), we collect **anonymized transition data**:

- Audio fingerprint pairs (which tracks were played in sequence)
- Transition metadata (BPM difference, key relationship, energy delta)
- Timestamp (rounded to the nearest day for anonymization)

**We do NOT collect**: track titles, artist names, file paths, playlist names, or any other personally identifiable information in community data.

### 2.4 Subscription Data

If you subscribe to AI DJ Assist Pro, Stripe (our payment processor) handles all billing information. We store only:

- Your Stripe customer ID (an opaque identifier)
- Subscription tier (Free or Pro)
- Subscription status (active, canceled, trial)

We do NOT store credit card numbers, bank details, or other financial information. See [Stripe's Privacy Policy](https://stripe.com/privacy) for how Stripe handles your payment data.

## 3. How We Use Your Data

| Data | Purpose | Legal Basis |
|------|---------|-------------|
| Account data | Authentication, account management | Contract performance |
| Local analysis data | Track recommendations, mix suggestions | Legitimate interest (core functionality) |
| Community transition data | Improve recommendations for all users | Consent (opt-in) |
| Subscription data | Feature access, billing management | Contract performance |

## 4. The Give-to-Get Model

AI DJ Assist uses a "Give-to-Get" model for community features:

- **What you give**: Anonymized transition data from your mix history (which tracks you played in sequence, with no identifying information)
- **What you get**: Access to community-enhanced recommendations powered by aggregated data from all contributing DJs
- **It is optional**: You can use AI DJ Assist without contributing. Local analysis and basic recommendations work fully offline
- **You can withdraw at any time**: See Section 7

### 4.1 Anonymization Process

Before any data leaves your device:

1. Track identifiers are converted to audio fingerprints (content-based, not file-based)
2. All personal metadata is stripped (no file paths, artist names, or playlist info)
3. Timestamps are rounded to the nearest day
4. Data is transmitted over encrypted connections (TLS 1.3)

No individual user's listening habits can be reconstructed from the community dataset.

## 5. Data Storage and Security

- **Local data**: Stored in an encrypted SQLite database on your device, managed by the operating system's file permissions
- **Community data**: Stored in a PostgreSQL database hosted on [INSERT CLOUD PROVIDER] with encryption at rest (AES-256) and in transit (TLS 1.3)
- **Authentication tokens**: Access tokens are short-lived (15 minutes). Refresh tokens are stored as bcrypt hashes in our database and in your operating system's secure keychain
- **Infrastructure**: All backend services run in [INSERT REGION] with SOC 2-compliant hosting

## 6. Data Sharing

We do NOT sell, rent, or share your personal data with third parties, except:

- **Stripe**: For payment processing (Pro subscribers only)
- **Google/Apple**: For authentication (OAuth sign-in only)
- **Law enforcement**: Only when required by valid legal process

Community transition data is aggregated and anonymized -- individual contributions cannot be traced back to specific users.

## 7. Your Rights

### 7.1 Right to Access

You can view all data associated with your account at any time through the app's Settings > Privacy section.

### 7.2 Right to Deletion (Data Withdrawal)

You can delete your account and all associated data:

1. Open AI DJ Assist > Settings > Account > Delete Account
2. This will:
   - Immediately revoke all active sessions
   - Permanently delete your account data (email, display name, OAuth link)
   - Remove your user association from any community contributions
   - Cancel any active subscription
3. Community transition data you previously contributed will be **retained in anonymized form** (audio fingerprint pairs only, with no link to your account). This data cannot be traced back to you.
4. Local data on your device is not affected -- you can delete it by removing the application.

### 7.3 Right to Withdraw Consent

You can stop contributing community data at any time:

1. Open AI DJ Assist > Settings > Privacy > Community Sharing
2. Toggle off "Share anonymized mix history"
3. No further data will be uploaded

Previously contributed data remains in the anonymized community dataset (see 7.2 for full deletion).

### 7.4 Right to Data Portability

You can export your local data (analysis results, preferences, mix history) in standard formats (JSON, CSV) through Settings > Data > Export.

## 8. Data Retention

| Data Type | Retention Period |
|-----------|-----------------|
| Account data | Until account deletion |
| Local analysis data | Until app removal (user-controlled) |
| Community transition data | Indefinite (anonymized, no PII) |
| Authentication sessions | 30 days (auto-expire) |
| Server logs | 90 days |

## 9. Children's Privacy

AI DJ Assist is not intended for users under 16 years of age. We do not knowingly collect data from children.

## 10. Changes to This Policy

We will notify you of material changes to this Privacy Policy via:

- In-app notification
- Email (if you have an account)

Continued use of the Service after notification constitutes acceptance of the updated policy.

## 11. Contact

For privacy inquiries or data requests:

- Email: [INSERT PRIVACY EMAIL]
- In-app: Settings > Help > Privacy Inquiry

## 12. Jurisdiction

This Privacy Policy is governed by the laws of [INSERT JURISDICTION]. For users in the European Economic Area, we comply with GDPR requirements as described in this policy.
