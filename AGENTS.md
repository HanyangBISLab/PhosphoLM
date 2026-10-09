# Reproducing PhosphoLM results

Use this guide when working on experiments in this repository. Keep commands and documentation simple. Use the existing scripts and supplied folds.

## Setup and inputs

- Run commands from the repository root.
- Follow [README.md](README.md) to create and activate the `ptm_language` Conda environment using `environment.yml`.
- Download [PhosphoLM.zip](https://drive.google.com/file/d/15ydKegyDKyzuO8fjJZ1RmCYmtgdHnKN7/view?usp=sharing) and merge its `PhosphoLM/` folder into the repository.
- Use `models/pretrained_phosphoLM/` for the pretrained model and tokenizer.
- Keep dataset folds in `datasets/<task>/folds/`. Phosphosite folds are in `datasets/dbptm/folds/`.
- Use supplied embedding caches for PPI and for PTM-Mamba disease and druggability. PhosphoLM and ESM-2 disease/druggability training can generate missing embeddings.
- Follow the [PTM-Mamba repository](https://github.com/programmablebio/ptm-mamba) for PTM-Mamba-specific setup and workflows. Use this repository's `ptm_mamba_` scripts for the supported downstream experiments.
- Use a suitable NVIDIA GPU for training. Token training uses BF16; MLM pretraining also uses FlashAttention.

## Choose the required workflow

Run only the stages needed for the requested result:

1. With predictions available, run the comparison plot script.
2. With downstream checkpoints and required embeddings available, run inference, then plot.
3. To train downstream models, run training, inference where needed, then plot.
4. Run tokenizer training and MLM pretraining only when reproducing pretraining.

Keep existing research artifacts available for comparison. Use alternate output paths where supported or a separate working copy for fresh runs. Keep verification scripts and temporary reports outside the repository.

## Downstream experiments

Script names follow this pattern:

```bash
python <model>_<task>_kfold_training.py
python <model>_<task>_inference.py
```

Replace the placeholders using this table:

| Task | Script task name | Folds | Available model prefixes |
|---|---|---:|---|
| Phosphosite | `phosphosite` | 0–9 | `phosphoLM`, `esm2_650M` |
| Functional phosphosite | `funcphos` | 0–4 | `phosphoLM`, `esm2_650M` |
| Disease association | `disease` | 0–4 | `phosphoLM`, `esm2_650M`, `ptm_mamba` |
| Druggability | `druggable` | 0–4 | `phosphoLM`, `esm2_650M`, `ptm_mamba` |
| PPI | `ppi` | 0–4 | `phosphoLM`, `esm2_650M`, `ptm_mamba` |

For example:

```bash
python phosphoLM_phosphosite_kfold_training.py
python phosphoLM_phosphosite_inference.py
```

- Token training runs all configured folds and saves the selected checkpoint in `models/<model>_<task>/fold_<n>/final_model/`.
- Disease, druggability and PPI training also generate test predictions. Their inference scripts use cached test embeddings.
- Generate missing ESM-2 PPI caches with `python esm2_650M_ppi_save_embeddings.py`.
- `phosphoLM_ppi_save_embeddings.py` requires existing site embeddings and a destination supplied through `--embedding-root`; it recomputes partner embeddings.
- Obtain PTM-Mamba phosphosite and functional-phosphosite predictions externally for the complete comparison.
- Preserve the supplied folds, labels, token mapping, pooling, loss normalization and checkpoint-selection behavior. Downstream focal loss uses scalar alpha `0.85`, gamma `2.5`, and initial or peak learning rate `1e-4`.

## Comparison figures

```bash
python plot_comparision.py
```

For a single task, use `--tasks disease`, `druggable`, `phosphorylation`, `functional` or `ppi`.

Predictions belong in `output/<task>/<model>/predictions/fold_<n>_test.<extension>`:

- Use `phosphosite` and `funcphos` as the token-task folder names, with JSON predictions.
- Use CSV predictions for disease, druggability and PPI.
- Model folders are `phosphoLM`, `esm2_650M` and `ptm_mamba`, except PTM-Mamba phosphosite predictions use `ptm-mamba`.

A complete comparison reads 90 prediction files and produces 15 PNGs under `figures/<plot-task>/`: ROC curves, precision–recall curves and specificity metric bars for each task.

Reuse the loaders and metric functions in `plot_comparision.py`. AUPRC uses trapezoidal precision–recall integration. Average folds equally and use sample standard deviation for error bars. Specificity operating points must meet at least 90% or 95% specificity. Preserve the existing threshold-selection rules.

## Token and distance analyses

Run these stages in order, reusing available results:

```bash
python phosphosite_proximal_token_analysis.py
conda run -n phospholm_pymol python calculate_phosphosite_token_distances.py
python plot_phosphosite_token_distance_heatmap.py
python plot_phosphosite_sequence_distance_heatmap.py
```

Create the separate PyMOL environment using the command in the README. Structural analysis uses `datasets/pdb/SEP_TPO_PTR_PDBs.json`; Pfam annotation uses `datasets/pfam/pdb_pfam_mapping.txt`.

- Selected tokens: `analysis/phosphosite_proximal_token_analysis/selected_tokens.csv`.
- Structural results: `analysis/phosphosite_token_distance/`.
- Sequence results: `analysis/phosphosite_sequence_distance/`.
- Figures: the corresponding folders under `figures/`.

Preserve hidden-state indices 0–32 and the structural token selection from states 15–32. The proximal similarity window uses emitted-token positions; the sequence-distance plot uses amino-acid residue distances. Reuse structural caches only with their corresponding token selection and inputs.

## Optional pretraining and exports

```bash
python tokenizer_training.py
accelerate launch mlm_training.py
```

These require `datasets/phosphopeptides/phosphopeptides.txt` and `datasets/uniref50/uniref50.fasta`. Outputs are `output/phospholm_tokenizer/` and `output/mlm_pretraining/`. MLM resumes the latest checkpoint automatically. Downstream scripts expect the chosen pretrained model and tokenizer in `models/pretrained_phosphoLM/`.

Use `python phosphosite_save_embeddings.py --model both` to export PhosphoLM and ESM-2 phosphosite embeddings.
