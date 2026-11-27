# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this repository:

1. **Do NOT open a public GitHub issue** for undisclosed vulnerabilities.
2. Submit a private report via GitHub Security Advisories (`Security -> Advisories -> Report a vulnerability`).
3. Include affected file paths, line numbers, reproduction steps, and estimated severity.
4. Initial acknowledgement will be provided within **48 hours**.

## Security Controls

- **Secret Hygiene**: `.env`, `.env.*`, and private key artifacts are excluded via `.gitignore`. Zero hardcoded credentials exist in source or tests.
- **Automated Vulnerability Management**: Weekly Dependabot security updates (`semver-major` ignored), Semgrep SAST (`p/python`), and Trivy container scanning in CI.
- **API Input & Boundary Hardening**: Strict Pydantic v2 field validation (`MAX_PREDICT_BATCH = 500`), explicit CORS middleware configuration, and centralized error handling masking stack traces.
- **Container Hardening**: Multi-stage non-root Docker runtime (`appuser`, UID 10001).
