# RECITE: Recitation-Augmented Language Models

See ["Recitation-Augmented Language Models"](https://openreview.net/forum?id=-cqvvvb-NkI) (ICLR 2023) for the paper associated with this codebase.

<p align="center">
  <img src="https://github.com/Edward-Sun/RECITE/blob/main/recite_illustration.png?raw=true" alt="RECITE Illustration"/>
</p>

## Setup

```bash
conda env create -f environment.yml
conda activate recite
```

## Reproduction

### Natural Questions

Direct:
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"

python -u codex_experiments/run_hotpot_qa.py \
  --num_examples 1024 \
  --inference_scheme "direct" \
  --batch_size 8 \
  --verbose
```

RECITE:
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"

python -u codex_experiments/run_hotpot_qa.py \
  --num_examples 1024 \
  --self_consistency_k 20 \
  --inference_scheme "recite" \
  --batch_size 8 \
  --verbose
```

Direct + BM25 (start the BM25 server from `bm25_retrieval_server` first):
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"

python -u codex_experiments/run_natural_questions.py \
  --num_examples 1024 \
  --inference_scheme "with_bm25_context" \
  --batch_size 8 \
  --verbose
```

Direct + Ground-truth Context:
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"

python -u codex_experiments/run_natural_questions.py \
  --num_examples 1024 \
  --inference_scheme "with_ground_truth_context" \
  --batch_size 8 \
  --verbose
```

### TriviaQA

Direct:
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"

python -u codex_experiments/run_trivia_qa.py \
  --num_examples 1024 \
  --inference_scheme "direct" \
  --batch_size 8 \
  --verbose
```

RECITE:
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"

python -u codex_experiments/run_trivia_qa.py \
  --num_examples 1024 \
  --inference_scheme "recite" \
  --batch_size 8 \
  --verbose
```

### HotpotQA

Direct:
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"

python -u codex_experiments/run_hotpot_qa.py \
  --num_examples 1024 \
  --inference_scheme "direct" \
  --batch_size 8 \
  --verbose
```

RECITE:
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"

python -u codex_experiments/run_hotpot_qa.py \
  --num_examples 1024 \
  --self_consistency_k 20 \
  --inference_scheme "recite" \
  --batch_size 8 \
  --verbose
```

Chain-of-thought:
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"

python -u codex_experiments/run_hotpot_qa.py \
  --num_examples 1024 \
  --inference_scheme "cot" \
  --batch_size 1 \
  --verbose
```

## Reference

If you found this codebase useful, please consider citing the paper:

```
@inproceedings{
    sun2023recitationaugmented,
    title={Recitation-Augmented Language Models},
    author={Zhiqing Sun and Xuezhi Wang and Yi Tay and Yiming Yang and Denny Zhou},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=-cqvvvb-NkI}
}
```
