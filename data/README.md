# Data Directory

This directory is a placeholder for dataset-related documentation.

## Dataset: RGIIIT

The raw dataset (`RGIIIT/`) and cleaned dataset (`RGIIIT_clean/`) are **not tracked by Git** due to their size. They must be obtained separately.

### How to obtain the dataset

1. Place the raw `RGIIIT/` folder in the project root.
2. Run the cleaning pipeline to generate `RGIIIT_clean/`:

```bash
python scripts/cleaning/clean_and_split_dataset.py
```

3. Validate the cleaned dataset:

```bash
python scripts/validation/validate_clean_dataset.py
```

### Dataset metadata (committed to Git)

The following metadata files are committed and available in `dataset_cleaning/`:

| File | Description |
|------|-------------|
| `split_manifest.csv` | Per-image manifest with split assignments |
| `cleaning_log.csv` | Log of all cleaning actions taken |
| `subject_split_map.json` | Subject → split mapping |
| `dataset_config.json` | Full pipeline configuration |

Audit reports are available in `dataset_audit/reports/`.
