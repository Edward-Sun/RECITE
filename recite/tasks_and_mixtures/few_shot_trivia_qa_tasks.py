"""few-shot HotpotQA open tasks."""

import functools

import seqio
import t5.data
import tensorflow as tf
import tasks_and_mixtures.preprocessors as preprocessors
import tasks_and_mixtures.gpt2_vocab as gpt2_vocab


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

for n_shots in [0, 1, 2, 5, 8, 16, 32, 64]:
  task_suffix = TASK_SUFFIX
  output_features = GPT2_FEATURES

  for eval_on_fixed_exemplars in [False, True]:
    fixed_suffix = "_fixed" if eval_on_fixed_exemplars else ""
    seqio.TaskRegistry.add(
        f"learning_to_recite:trivia_qa_rc_open_{n_shots}shot{fixed_suffix}_{task_suffix}",
        source=seqio.experimental.FewshotDataSource(
            original_source=seqio.TfdsDataSource(
                tfds_name="trivia_qa/rc.nocontext:1.1.0",
                splits={
                    "train": "train",
                    "validation": "validation[:1024]",
                    "test": "validation",
                }),
            num_shots=n_shots,
            train_preprocessors=[
                functools.partial(
                    preprocessors.trivia_qa_open,
                    prefix="",
                    suffix="\n\n",
                    max_tokens=16)
            ],
            eval_preprocessors=[
                functools.partial(
                    preprocessors.trivia_qa_open,
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
        metric_fns=[t5.evaluation.metrics.trivia_qa],
    )

