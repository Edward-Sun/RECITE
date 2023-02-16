"""few-shot HotpotQA open tasks."""

import functools
import re
import string

import datasets
import seqio
import t5.data
from t5.evaluation import qa_utils
import tensorflow as tf

import tasks_and_mixtures.preprocessors as preprocessors
import tasks_and_mixtures.gpt2_vocab as gpt2_vocab


def hotpotqa_normalize_answer(s):
  """Adpoted from https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py."""
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def hotpot_qa_metric(targets, predictions):
  """Computes HotpotQA metrics, addopted from t5.evaluation.metrics.trivia_qa.

  Args:
    targets: list of lists of strings
    predictions: list of strings

  Returns:
    dict with score_key: squad score across all targets and predictions
  """
  targets = [[hotpotqa_normalize_answer(t) for t in u] for u in targets]
  predictions = [hotpotqa_normalize_answer(p) for p in predictions]
  return qa_utils.qa_metrics(targets, predictions)


def fewshot_cbqa_postprocessor_answers(answer, is_target=False, **kwargs):
  if not is_target:
    try:
      answer = answer.split("\n\n")[0]
    except Exception as _:  # pylint: disable=broad-except
      answer = ""
  return t5.data.postprocessors.qa(answer, is_target=is_target, **kwargs)


fewshot_cbqa_preprocessor = functools.partial(
    seqio.experimental.fewshot_preprocessor,
    inputs_prefix="Q: ",
    targets_prefix="\n\nA: ",
    example_separator="")

TASK_SUFFIX = "lm_gpt2"
GPT2_FEATURES = {
    "inputs": seqio.Feature(gpt2_vocab.GPT2Vocabulary(), add_eos=False, dtype=tf.int32),
    "targets": seqio.Feature(gpt2_vocab.GPT2Vocabulary(), add_eos=False, dtype=tf.int32),
}


def get_hq_tfds(split="train", shuffle_files=False, seed=None):
  def dataset_generator():
    limit = None
    if split == "validation":
      real_split = split
      limit = 1024
    elif split == "test":
      real_split = "validation"
    else:
      real_split = split

    hf_hotpot_qa_ds = datasets.load_dataset("hotpot_qa", "distractor", split=real_split)
    if shuffle_files:
      hf_hotpot_qa_ds = hf_hotpot_qa_ds.shuffle(seed=seed)
    cnt = 0
    for ex in hf_hotpot_qa_ds:
      del ex["context"]
      yield ex
      cnt += 1
      if cnt == limit:
        break

  ds = tf.data.Dataset.from_generator(
      dataset_generator,
      output_signature={
          "id": tf.TensorSpec(shape=(), dtype=tf.string),
          "question": tf.TensorSpec(shape=(), dtype=tf.string),
          "answer": tf.TensorSpec(shape=(), dtype=tf.string),
          "type": tf.TensorSpec(shape=(), dtype=tf.string),
          "level": tf.TensorSpec(shape=(), dtype=tf.string),
          "supporting_facts": {
              "sent_id": tf.TensorSpec(shape=(None,), dtype=tf.int32),
              "title": tf.TensorSpec(shape=(None,), dtype=tf.string),
          }
      })
  return ds


for n_shots in [0, 1, 2, 4, 5, 32, 64]:
  task_suffix = TASK_SUFFIX
  output_features = GPT2_FEATURES

  for eval_on_fixed_exemplars in [False, True]:
    fixed_suffix = "_fixed" if eval_on_fixed_exemplars else ""
    seqio.TaskRegistry.add(
        f"learning_to_recite:hotpot_qa_open_{n_shots}shot{fixed_suffix}_{task_suffix}",
        source=seqio.experimental.FewshotDataSource(
            original_source=seqio.FunctionDataSource(
                dataset_fn=get_hq_tfds,
                splits=("train", "validation", "test"),
            ),
            num_shots=n_shots,
            train_preprocessors=[
                functools.partial(
                    preprocessors.hotpot_qa_open,
                    prefix="",
                    suffix="\n\n",
                    max_tokens=16)
            ],
            eval_preprocessors=[
                functools.partial(
                    preprocessors.hotpot_qa_open,
                    prefix="",
                    suffix="\n\n")
            ],
            eval_on_fixed_exemplars=eval_on_fixed_exemplars,
        ),
        preprocessors=[
            fewshot_cbqa_preprocessor,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            preprocessors.append_eos_after_trim_at_front,
        ],
        postprocess_fn=fewshot_cbqa_postprocessor_answers,
        output_features=output_features,
        metric_fns=[hotpot_qa_metric],
    )
