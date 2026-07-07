# Few-shot Determinants

This repository contains the code, configuration files, processed experimental data, and manuscript-ready outputs for the study:

**Few-shot cross-domain sentiment classification performance determinants: pretraining level, domain proximity, and methodological stability**

The project investigates how few-shot cross-domain sentiment classification performance changes across model type, source-target domain relation, shot level, and repeated random seeds.

## Repository DOI

A DOI can be assigned to this repository through a research archive such as Zenodo. After the DOI is issued, replace the placeholder below with the final DOI.

**Repository DOI:** `10.5281/zenodo.21236009`

## Citation

If you use this repository, please cite it as:

```text
Kapar, F. (2026). Few-shot Determinants: Code and outputs for few-shot cross-domain sentiment classification experiments [Software and data]. Zenodo. https://doi.org/10.5281/zenodo.21236009
```

BibTeX:

```bibtex
@misc{kapar2026fewshotdeterminants,
  author       = {Kapar, Fırat},
  title        = {Few-shot Determinants: Code and outputs for few-shot cross-domain sentiment classification experiments},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21236009},
  url          = {https://doi.org/10.5281/zenodo.21236009}
}
```

## Study overview

The study focuses on three determinants of few-shot cross-domain sentiment classification performance:

1. **Pretraining level**
   - Comparison of a classical non-pretrained baseline and pretrained transformer models.

2. **Domain proximity**
   - Analysis of source-target domain similarity using lexical, label-distribution, and combined similarity measures.

3. **Methodological stability**
   - Repeated experiments across multiple random seeds and shot levels.

The experiments are conducted on four product-review domains:

- `books`
- `dvd`
- `electronics`
- `kitchen & housewares`

All source-target pairs with different domains are evaluated, resulting in 12 directional transfer settings.

## Models

The repository compares three model lines:

| Model | Role in the study |
|---|---|
| TF-IDF + Logistic Regression | Classical BOW baseline |
| DistilBERT | Lightweight pretrained transformer model |
| RoBERTa-base | Stronger pretrained transformer model |

The few-shot settings use class-balanced target-domain support sets with:

- 4 examples per class
- 8 examples per class
- 16 examples per class

## Repository structure

```text
configs/                         YAML configuration files
data/                            Raw, interim, and processed data files
docs/                            Project notes and structure documentation
notebooks/                       Optional exploratory notebooks
outputs/                         Raw experiment outputs and comparison outputs
reports/figures/                 Final figures used in the manuscript
reports/tables/                  Final tables used in the manuscript
scripts/                         Entry-point scripts for experiments and analyses
src/fewshot_determinants/        Core Python package
tests/                           Basic tests
```

## Environment setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Check the environment:

```bash
python scripts/00_check_env.py
```

## Data preparation

The processed dataset used by the experiments is stored at:

```text
data/processed/amazon_multidomain_sentiment_rawtext.csv
```

To regenerate the processed data file, run:

```bash
python scripts/04_inspect_hf_multidomain_source.py
python scripts/05_prepare_hf_multidomain_dataset.py
python scripts/06_dataset_summary.py
```

The processed file contains the following core columns:

| Column | Description |
|---|---|
| `text` | Review text used as model input |
| `label` | Binary sentiment label |
| `domain` | Product-review domain |
| `split` | Train, validation, or test split |

Sentiment labels are defined as follows:

- 1-star and 2-star reviews are mapped to the negative class.
- 4-star and 5-star reviews are mapped to the positive class.
- 3-star reviews are removed as neutral or ambiguous cases.

## Running the experiments

### 1. BOW baseline

```bash
python scripts/09_run_bow_repeats_shots.py
```

Final BOW outputs:

```text
outputs/runs/20260422_230251_bow_repeats_shots/
```

### 2. DistilBERT transfer matrix

```bash
python scripts/16_run_distilbert_transfer_matrix_shots.py
```

Final DistilBERT outputs:

```text
outputs/runs/20260423_155145_distilbert_transfer_matrix_shots/
```

### 3. RoBERTa-base transfer matrix

```bash
python scripts/19_run_roberta_transfer_matrix_shots.py
```

Final RoBERTa-base outputs:

```text
outputs/runs/20260423_232310_roberta_transfer_matrix_shots/
```

## Statistical comparisons

Shot-level paired model comparisons can be generated with:

```bash
PYTHONPATH=src python scripts/20_compare_model_pair_shots.py --config configs/comparison_bow_vs_roberta.yaml
PYTHONPATH=src python scripts/20_compare_model_pair_shots.py --config configs/comparison_distilbert_vs_roberta.yaml
```

Existing comparison outputs include:

```text
outputs/comparisons/20260424_073656_comparison_bow_vs_roberta/
outputs/comparisons/20260424_073722_comparison_distilbert_vs_roberta/
```

The final manuscript-ready paired comparison table is also stored at:

```text
reports/tables/table_4_paired_model_comparisons.csv
```

## Domain similarity analysis

Domain similarity analysis is configured through:

```text
configs/analysis_domain_similarity.yaml
```

Run:

```bash
python scripts/17_analyze_domain_similarity.py
```

Final domain similarity outputs:

```text
outputs/analyses/20260630_121845_domain_similarity_analysis/
```

The manuscript-ready summary table is stored at:

```text
reports/tables/table_6_domain_similarity_correlations.csv
```

## Final outputs used in the manuscript

The final manuscript results are based on the following outputs.

### Model runs

```text
outputs/runs/20260422_230251_bow_repeats_shots/
outputs/runs/20260423_155145_distilbert_transfer_matrix_shots/
outputs/runs/20260423_232310_roberta_transfer_matrix_shots/
```

### Model comparisons

```text
outputs/comparisons/20260424_073656_comparison_bow_vs_roberta/
outputs/comparisons/20260424_073722_comparison_distilbert_vs_roberta/
reports/tables/table_4_paired_model_comparisons.csv
```

### Domain similarity analysis

```text
outputs/analyses/20260630_121845_domain_similarity_analysis/
reports/tables/table_6_domain_similarity_correlations.csv
```

## Manuscript tables and figures

Final tables and figures used in the manuscript are stored under:

```text
reports/tables/
reports/figures/
```

Current manuscript-ready assets:

```text
reports/tables/table_1_dataset_summary.csv
reports/tables/table_3_model_summary.csv
reports/tables/table_4_paired_model_comparisons.csv
reports/tables/table_5_shot_summary.csv
reports/tables/table_6_domain_similarity_correlations.csv
reports/figures/figure_1_experimental_design.png
reports/figures/figure_1_experimental_design.pdf
reports/figures/figure_2_shot_performance.png
reports/figures/figure_2_shot_performance.pdf
```

## Reproducibility notes

- All final experiments use fixed random seeds.
- The BOW baseline uses TF-IDF features and Logistic Regression.
- Transformer experiments use the same source-target transfer structure and shot settings.
- Validation data is used for training monitoring only.
- Final evaluation is performed on the target-domain test split.
- Older smoke-test and intermediate folders may remain in `outputs/` for development history. They are not used in the final manuscript unless explicitly listed above.

## License

Add the project license here before public archiving.

## Contact

For questions about the repository, contact:

**Fırat Kapar**
