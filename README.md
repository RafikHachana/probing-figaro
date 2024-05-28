# FIGARO Interpretability study

This code is built on top of the original FIGARO repository. The main code additions are in the following files:
- `models/probe.py`
- `models/seq2seq.py`
- `train_probe.py`

In order to run a probe experiment, you will first need to generate a training dataset for the probe:

```bash
!cd figaro && MODEL=figaro CHECKPOINT=/figaro-expert.ckpt \
ROOT_DIR=/lmd_clean/ \
ALTER_DESCRIPTION=False \
MAX_ITER=256 \
BATCH_SIZE=30 \
OUTPUT_DIR=/output/ \
MAX_N_FILES=50000 \
DATASET_PATH=/probe_dataset \
python src/generate_probe_dataset.py
```

Then train the probe on a certain token category using the generated dataset:

```bash
!cd figaro && \
DATASET_PATH=/probe_dataset \
TOKEN_CLASS_OF_INTEREST=chords \
python src/train_probe.py
```



The code for attention visualization can be found in `Attention_visualization.ipynb` 
