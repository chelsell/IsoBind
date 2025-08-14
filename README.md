# IsoBind

IsoBind is a pharmacological profile similarity and visualization toolkit.  
It ingests chemical–target activity data, deduplicates chemical identifiers, aggregates target profiles by parent compound, and computes **Rank-Biased Overlap (RBO)** distances between compounds for embedding and clustering.

---

## Features

- **ChEMBL deduplication** using salts/stereo/brand merges (union–find buckets).
- **Flexible profile aggregation** from compound → canonical parent bucket.
- **Tie-aware RBO** (SIGIR 2024 implementation) for ranked target list comparison.
- **2D/3D embeddings** via SMACOF for visualization.
- **QC sentinels** for merge sanity and nearest-neighbor inspection.

---

## Installation

### Requirements

- Python ≥ 3.9  
- Dependencies: see `requirements.txt`
- sigir2024 rbo. The most visible pypi package is broken.
git clone https://github.com/cosrisi/sigir2024-rbo.git  # or your fork
and either install it into your environment or pass its path to --rbo-path.
### Install (editable mode)

```bash
git clone git@github.com:<your-username>/<new-repo-name>.git
cd <new-repo-name>
pip install -e .
```

### Quickstart
1. Aggregate profiles by deduplication buckets
python isobind_pipeline.py aggregate \
  --profiles profiles.csv \
  --members dedup_bucket_membership.csv \
  --out bucket_profiles.csv

# 2. Build ranked target lists
python isobind_pipeline.py ranks \
  --bucket-profiles bucket_profiles.csv \
  --out ranked_targets.csv

# 3. Compute RBO distances (tie-aware SIGIR impl)
python isobind_pipeline.py rbo \
  --ranked ranked_targets.csv \
  --out isobind_distances.csv \
  --p 0.98 \
  --rbo-path /path/to/sigir2024-rbo/

# 4. Embed in 2D
python isobind_pipeline.py embed \
  --dist isobind_distances.csv \
  --out isobind_embedding2d.csv

# 5. Run QC
python isobind_pipeline.py qc \
  --dist isobind_distances.csv \
  --canon canonical_map.csv

### File Descriptions
- isobind_pipeline.py – Main CLI pipeline.

- requirements.txt – Python dependencies.

- LICENSE – Licensing terms.

- README.md – This documentation.

### License

This repository is licensed under a custom open-source license:

    Free for non-commercial use, modification, and distribution with attribution.

    Commercial use requires separate licensing from the authors.
See LICENSE for details.


