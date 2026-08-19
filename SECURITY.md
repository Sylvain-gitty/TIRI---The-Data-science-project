# Security Policy

## What this repository is

TIRI is a **research and analysis repository**: notebooks, analysis scripts, written reports,
and a labelled dataset. It is not a deployed service, it exposes no network endpoints, it
handles no user accounts, and it processes no personal data.

So the realistic security surface here is narrow. It is not zero, and these are the things
worth reporting:

- **A leaked credential.** An API key, token, or `.env` fragment committed to the repository
  or embedded in a notebook's saved output. `.env` is gitignored and the repository has been
  scanned, but scans are not proofs.
- **Personal or sensitive data in the shipped dataset.** `data/raw/*.jsonl` contains public
  academic metadata (titles, abstracts, venues, author names as published) plus an analyst
  relevance label. If you find anything in there that should not be public, that is a
  reportable problem and we want to know.
- **A supply-chain issue** in a dependency pinned by `requirements.txt` or
  `requirements-lock.txt`.
- **Unsafe code that executes on input**, for example a deserialisation path that would run
  arbitrary code when reading a data file.

## Reporting

**Please do not open a public issue for a credential leak or a data-exposure problem.**

Use GitHub's private vulnerability reporting on this repository
(**Security** → **Report a vulnerability**), which opens a channel visible only to the
maintainers. If that is unavailable, contact a maintainer through their GitHub profile
([@Sylvain-gitty](https://github.com/Sylvain-gitty)) and ask for a private channel before
sending details.

For anything that is *not* sensitive — a pinned dependency with a published CVE, say — a
normal public issue is fine and easier to track.

## What to expect

This is a small project maintained by two people alongside other work, so please calibrate
accordingly:

- We will acknowledge a report within **7 days**.
- For a confirmed credential leak we will rotate and revoke first, then fix the history.
- For a confirmed data-exposure problem we will remove the data first and discuss the fix
  afterwards.
- We will credit you in the fix unless you would rather we did not.

## Out of scope

- Findings against the `.venv/` directory, which is gitignored and is your local environment,
  not this repository's code.
- Vulnerabilities in third-party services this code can optionally call (OpenRouter, Modal,
  academic metadata APIs). Report those to the service.
- The absence of authentication, rate limiting, or input sanitisation in analysis scripts
  intended to be run locally by their author against local files.
