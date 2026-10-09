from pathlib import Path

import numpy as np

ESM2_CHECKPOINT = "facebook/esm2_t33_650M_UR50D"
ESM2_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
PHOSPHOLM_CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "models/pretrained_phosphoLM"
)


def save_ppi_cache(path, source, sites, binders):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sites = np.asarray(sites, dtype=np.float32)
    binders = np.asarray(binders, dtype=np.float32)
    np.savez_compressed(
        path,
        embeddings=np.concatenate((sites, binders), axis=1),
        site_embeddings=sites,
        binder_embeddings=binders,
        labels=source.labels.astype(np.int64, copy=False),
    )
