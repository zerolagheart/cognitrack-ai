# Security

## Reporting

Please report vulnerabilities privately to the maintainers (do not open a public issue containing exploit details or any data).

## Threat model (prototype scope)

| Asset | Threats | Controls |
|---|---|---|
| Participant-level research data | Unauthorised access, re-identification | Role-based API keys (researcher role for participant data); pseudonyms only; identity links separated and never exposed; audit log of participant-level access |
| Uploaded audio | Malicious files, path traversal, storage exhaustion | Extension allow-list; server-generated filenames (client name never used in paths); streamed size limit (413); decode check before acceptance; deletion after processing by default |
| API | Injection, abuse | Pydantic validation; SQLAlchemy parameterised queries; restrictive CORS (GET/POST, explicit origins); `X-Content-Type-Options: nosniff`; error envelope without stack traces |
| Secrets | Leakage | Environment-only configuration (`.env` git-ignored); constant-time key comparison; production refuses to start without keys |
| Logs | PII leakage | Path-only request logs (no query strings or bodies); email and long-number redaction |
| Model artifacts | Tampering, derived-data leakage | Local registry; artifacts git-ignored; joblib bundles must only be loaded from trusted locations (pickle executes code) |

## Hardening checklist for any non-local deployment

* [ ] Set strong per-role API keys, or replace them with OIDC/SSO behind a reverse proxy.
* [ ] Serve over TLS only; set HSTS at the proxy.
* [ ] Change the PostgreSQL credentials, keep the DB unexposed, and enable encryption at rest.
* [ ] Move `identity_links` to a separately secured database or drop it.
* [ ] Keep `COGNITRACK_RETAIN_RAW_AUDIO=false` unless the protocol requires retention.
* [ ] Add rate limiting at the proxy and set log retention limits.
* [ ] Run dependency audits (`pip-audit`, `npm audit`) in CI.
* [ ] Complete a data protection impact assessment and obtain ethics/IRB approval before using real data.

## Known limitations

* API keys are static shared secrets, with no per-user identity.
* Background processing runs in the API process.
* The frontend stores an entered API key in `sessionStorage` (tab-scoped). A `VITE_API_KEY` baked into a static build is visible to anyone with access to the UI, so use only a viewer key there.
