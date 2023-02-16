"""few-shot Natural Questions open tasks."""

import functools

import seqio
import t5.data
import t5.evaluation.metrics
import tensorflow as tf

import tasks_and_mixtures.preprocessors as preprocessors
import tasks_and_mixtures.gpt2_vocab as gpt2_vocab
import tasks_and_mixtures.nq_section_chain_utils as sc_utils
from tasks_and_mixtures.prompts import RECITE_PROMPT, SHORT_ANSWER_PROMPT


def fewshot_cbqa_postprocessor_answers(answer, is_target=False, **kwargs):
  if not is_target:
    try:
      answer = answer.split("\n\n")[0]
    except Exception as _:  # pylint: disable=broad-except
      answer = ""
  return t5.data.postprocessors.qa(answer, is_target=is_target, **kwargs)


def fewshot_section_chain_cbqa_postprocessor(
    answer, is_target=False, **kwargs):
  """Post processing for the section chain retrieval task.

  Args:
    answer: a string (prediction) or list of string (targets)
    is_target: bool
    **kwargs: other arguments

  Returns:
    a processed answer
  """
  if not is_target:
    try:
      answer = answer.split("\n")[0]
    except Exception as _:  # pylint: disable=broad-except
      pass

  return t5.data.postprocessors.qa(answer, is_target=is_target, **kwargs)


def long_answer_retrieve_and_answer(dataset: tf.data.Dataset):
  """Convert NQ dataset to question => passage + answer."""

  def nq_with_section_map(ex):
    return {
        "inputs":
            ex["inputs"],
        "targets":
            tf.strings.join([
                tf.strings.strip(
                    tf.strings.regex_replace(
                        ex["long_answer"], "[\n]+", "\n")),
                "\n\n",
                SHORT_ANSWER_PROMPT,
                ex["targets"],
                "\n\n\n"
            ]),
        "answers":
            ex["answers"]
    }

  dataset = dataset.map(
      nq_with_section_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


fewshot_cbqa_preprocessor = functools.partial(
    seqio.experimental.fewshot_preprocessor,
    inputs_prefix="Q: ",
    targets_prefix="\n\nA: ",
    example_separator="")

fewshot_section_chain_cbqa_preprocessor = functools.partial(
    preprocessors.fewshot_preprocessor,
    inputs_prefix="Question: ",
    targets_prefix=f"\n\n{RECITE_PROMPT}\n\nAnswer: ",
    example_separator="",
    prompt="")

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
        f"learning_to_recite:natural_questions_open_{n_shots}shot{fixed_suffix}_{task_suffix}",
        source=seqio.experimental.FewshotDataSource(
            original_source=seqio.TfdsDataSource(
                tfds_name="natural_questions_open:1.0.0",
                splits={
                    "train": "train",
                    "validation": "validation[:1024]",
                    "test": "validation",
                }),
            num_shots=n_shots,
            train_preprocessors=[
                functools.partial(
                    preprocessors.natural_questions_open,
                    prefix="",
                    suffix="\n\n",
                    max_tokens=16)
            ],
            eval_preprocessors=[
                functools.partial(
                    preprocessors.natural_questions_open,
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


for n_shots in [0, 1, 2, 5, 8, 16, 32, 64]:
  task_suffix = TASK_SUFFIX
  output_features = GPT2_FEATURES
  nq_max_tokens, nq_max_byte_length = None, None
  if n_shots > 5:
    nq_max_tokens = 16
    nq_max_byte_length = 512

  task_type = "long_answer"
  task_preprocessor = long_answer_retrieve_and_answer

  for eval_on_fixed_exemplars in [False, True]:
    fixed_suffix = "_fixed" if eval_on_fixed_exemplars else ""
    seqio.TaskRegistry.add(
        f"learning_to_recite:natural_questions_open_{task_type}_{n_shots}shot{fixed_suffix}_{task_suffix}",
        source=seqio.experimental.FewshotDataSource(
            original_source=seqio.TfdsDataSource(
                tfds_name="NaturalQuestions/default:0.1.*",
                splits={
                    "train": "train",
                    "validation": "validation[:1024]",
                    "test": "validation",
                }),
            num_shots=n_shots,
            train_preprocessors=[
                functools.partial(
                    sc_utils.natural_questions_with_section_chain,
                    drop_yes_no=True,
                    max_answers=1,
                    prefix="",
                    max_tokens=nq_max_tokens,
                    max_byte_length=nq_max_byte_length),
                task_preprocessor,
            ],
            eval_preprocessors=[
                functools.partial(
                    sc_utils.natural_questions_with_section_chain,
                    drop_yes_no=True,
                    max_answers=1,
                    prefix="",
                    max_tokens=nq_max_tokens,
                    max_byte_length=nq_max_byte_length),
                task_preprocessor,
            ],
            eval_on_fixed_exemplars=eval_on_fixed_exemplars,
        ),
        preprocessors=[
            fewshot_section_chain_cbqa_preprocessor,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            preprocessors.append_eos_after_trim_at_front,
        ],
        postprocess_fn=fewshot_section_chain_cbqa_postprocessor,
        output_features=output_features,
        metric_fns=[t5.evaluation.metrics.trivia_qa],
    )
