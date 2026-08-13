# Dataset and license boundary

## Source

LateSignal uses the Criteo Sponsored Search Conversion Log as its V1 dataset.
Criteo describes one tab-delimited row per product-ad click, 90 days of traffic, conversions within a 30-day window, and hashed click-time features.
The repository does not mirror or redistribute the archive.

The source page is <https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/>.
The configured archive URL and the observed planning-time byte count live in `configs/data.yaml`.
The source page's older compressed-size statement is informative but is not an artifact validation value.
The exact observed archive-member correction is documented in [source-artifact-review.md](source-artifact-review.md).

## Terms

The dataset is governed separately from this MIT-licensed code under `CC-BY-NC-SA-4.0`.
The terms include attribution, noncommercial use, and ShareAlike requirements.
Every user is responsible for reviewing the source terms and determining that the intended use is permitted.

`latesignal data fetch` fails before opening the URL unless `--accept-license` is present.
The command displays the dataset, license, source page, archive URL, local destination, and noncommercial restriction before network access.
It records the acknowledgement, timestamp, configuration hash, and code version under the ignored local data root.

## First-download trust review

The official source does not currently publish a SHA-256 beside the configured archive.
The first authorized download therefore uses a fail-closed trust-on-first-use process.

1. The archive streams to a newly created temporary file while SHA-256 and byte count are computed.
2. The configured byte count and predeclared archive-safety limits are enforced.
3. The exact member contract is enforced before the artifact is moved to its content-addressed local path.
4. The artifact remains marked untrusted and cannot be inspected or prepared.
5. The user reviews the displayed hash and artifact provenance, then repeats the printed command with `--review-sha256 HASH`.
6. LateSignal rehashes the retained artifact, repeats the safety inspection, and writes the local artifact lock.

If Criteo publishes an authoritative digest, it may be added to the authored configuration before download after review.
Any byte-count, digest, or member-list change stops the pipeline rather than silently replacing the artifact.

## Publication boundary

Raw rows, prepared rows, quarantine rows, checkpoints, and ordinary experiment artifacts remain ignored.
Public outputs are limited to code, authored configuration, manifests without row values, small synthetic fixtures, and aggregate evidence that complies with the source terms.
