# Security policy

## Supported versions

Security fixes are applied to the current development branch until the first stable release establishes a version policy.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose credentials, restricted row-level data, unsafe archive handling, or a truth-leakage path.
Contact the repository owner privately and include a minimal reproduction, affected revision, and expected impact.
Do not attach the Criteo archive, extracted rows, prepared rows, model checkpoints, or secrets.

## Data safety

The downloader rejects unsafe archive members before any artifact is accepted for use.
The normal experiment workflow performs no network access.
Only `latesignal data fetch` may access the configured public dataset URL.
