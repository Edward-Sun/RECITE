"""Experiments for TriviaQA.
"""

import argparse
import logging
import os
import functools

import seqio.feature_converters as f_converters
import tensorflow as tf

import codex_experiments.evaluation as evaluation
from codex_experiments.utils import predict_call, recite_predict_call, contextual_predict_call
from codex_experiments.utils import get_majority_voting
import tasks_and_mixtures.few_shot_trivia_qa_tasks as few_shot_trivia_qa_tasks  # pylint: disable=unused-import
from tasks_and_mixtures.prompts import RECITE_PROMPT, QA_PROMPT
from tasks_and_mixtures.prompts_for_trivia_qa import TRIVIAQA_FIXED_RECITE_PROMPT
from tasks_and_mixtures.prompts_for_trivia_qa import TRIVIAQA_FIXED_DIRECT_PROMPT

import openai


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_shots", type=int, default=5)
  parser.add_argument("--self_consistency_k", type=int, default=20)
  parser.add_argument("--evaluation_split", type=str, default="test")
  parser.add_argument("--input_length", type=int, default=1536)
  parser.add_argument("--target_length", type=int, default=128)
  parser.add_argument("--num_examples", type=int, default=1024)
  parser.add_argument("--inference_scheme", type=str, default="recite")

  parser.add_argument("--batch_size", type=int, default=20)
  parser.add_argument("--verbose", action="store_true")

  parser.add_argument("--openai_organization", type=str, default=None)
  parser.add_argument("--openai_api_key", type=str, default=None)
  args = parser.parse_args()

  if args.openai_organization is None:
    args.openai_organization = os.getenv("OPENAI_ORGANIZATION")
    if args.openai_organization is None:
      raise ValueError("Codex experiments must specify openai_organization")

  if args.openai_api_key is None:
    args.openai_api_key = os.getenv("OPENAI_API_KEY")
    if args.openai_api_key is None:
      raise ValueError("Codex experiments must specify openai_api_key")

  openai.organization = args.openai_organization
  openai.api_key = args.openai_api_key

  return args


def tq_input_processing_fn(inputs,
                           vocab,
                           exemplar_splitter="\n\n\n",
                           sentence_splitter="\n\n",
                           inference_scheme="direct",
                           fixed_prompt=True):
  decoder_input_tokens, decoder_causal_attention = inputs
  decoder_input_tokens = decoder_input_tokens[decoder_causal_attention]
  decoder_input_tokens = decoder_input_tokens[:-1]
  decoder_inputs = vocab.decode(decoder_input_tokens)
  new_examples = decoder_inputs.split(sentence_splitter)[-2:]

  if not fixed_prompt:
    raise NotImplementedError("Not implemented yet")

  if inference_scheme == "direct":
    decoder_inputs = exemplar_splitter.join([TRIVIAQA_FIXED_DIRECT_PROMPT] + new_examples)
    return decoder_inputs
  elif inference_scheme == "recite":
    qa_inputs = exemplar_splitter.join([TRIVIAQA_FIXED_DIRECT_PROMPT, sentence_splitter.join(new_examples)])
    new_examples = sentence_splitter.join([new_examples[0], RECITE_PROMPT, new_examples[1]])
    new_examples = new_examples.replace(f"{sentence_splitter}Q:", f"{sentence_splitter}Question:")
    new_examples = new_examples.replace(f"{sentence_splitter}A:", f"{sentence_splitter}Answer:")
    recite_inputs = exemplar_splitter.join([TRIVIAQA_FIXED_RECITE_PROMPT, new_examples])
    return recite_inputs, qa_inputs
  else:
    raise ValueError("Unknown inference scheme: %s" % inference_scheme)


def direct_answer_fn(dataset, task,
                     batch_size,
                     exemplar_splitter="\n\n\n",
                     sentence_splitter="\n\n",
                     verbose=False):
  """Answer a question directly or using contexts.

  Args:
    dataset: A tf.data.Dataset.
    task: A seqio.Task.
    batch_size: Batch size for evaluation.
    exemplar_splitter: The string used to split the exemplars.
    sentence_splitter: The string used to split the sentences within an exemplar.
    verbose: Whether to print the input and output.

  Returns:
    A list of tokenized answers.
  """

  results = []
  vocab = task.output_features["targets"].vocabulary
  dataset = dataset.batch(batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE)

  get_decoder_inputs = functools.partial(
      tq_input_processing_fn,
      vocab=vocab,
      exemplar_splitter=exemplar_splitter,
      sentence_splitter=sentence_splitter,
      inference_scheme="direct")

  for batch_id, ex in dataset.as_numpy_iterator():
    all_tokens = ex["decoder_target_tokens"]
    causal_attention = ex["decoder_causal_attention"] > 0
    qa_inputs = list(map(get_decoder_inputs, zip(all_tokens, causal_attention)))

    predictions = predict_call(qa_inputs)

    for example_id in range(batch_id.shape[0]):
      if verbose:
        logging.info("** QA INPUTS %d **" % batch_id[example_id])
        logging.info(qa_inputs[example_id])
        logging.info("** FINAL OUTPUTS %d **" % batch_id[example_id])
        logging.info(predictions[example_id])
      prediction_tokens = vocab.encode(predictions[example_id])
      results.append((batch_id[example_id], prediction_tokens))

  return results


def recite_and_answer_fn(dataset, task,
                         batch_size,
                         self_consistency_k,
                         exemplar_splitter="\n\n\n",
                         sentence_splitter="\n\n",
                         verbose=False):
  """Recite and answer a question.

  Args:
    dataset: A tf.data.Dataset.
    task: A seqio.Task.
    batch_size: Batch size for evaluation.
    self_consistency_k: The number of self-consistency steps.
    exemplar_splitter: The string used to split the exemplars.
    sentence_splitter: The string used to split the sentences within an exemplar.
    verbose: Whether to print the input and output.

  Returns:
    A list of tokenized answers.
  """

  results = []
  vocab = task.output_features["targets"].vocabulary
  dataset = dataset.batch(batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE)

  get_decoder_inputs = functools.partial(
      tq_input_processing_fn,
      vocab=vocab,
      exemplar_splitter=exemplar_splitter,
      sentence_splitter=sentence_splitter,
      inference_scheme="recite")

  for batch_id, ex in dataset.as_numpy_iterator():
    all_tokens = ex["decoder_target_tokens"]
    causal_attention = ex["decoder_causal_attention"] > 0
    task_inputs = list(map(get_decoder_inputs, zip(all_tokens, causal_attention)))
    recite_inputs = [_[0] for _ in task_inputs]
    qa_inputs = [_[1] for _ in task_inputs]

    recite_predictions = recite_predict_call(recite_inputs, self_consistency_k)

    if verbose:
      for example_id in range(batch_id.shape[0]):
        logging.info("** RECITE INPUTS %d **" % batch_id[example_id])
        logging.info(recite_inputs[example_id])
        logging.info("** RECITE OUTPUTS %d **" % batch_id[example_id])
        logging.info("\n\n".join([_[example_id] for _ in recite_predictions]))

    qa_predictions = contextual_predict_call(qa_inputs, recite_predictions)
    predictions = list(map(get_majority_voting, qa_predictions))

    for example_id in range(batch_id.shape[0]):
      if verbose:
        logging.info("** QA INPUTS %d **" % batch_id[example_id])
        logging.info(qa_inputs[example_id])
        logging.info("** VOTING %d **" % batch_id[example_id])
        logging.info(" ; ".join(qa_predictions[example_id]))
        logging.info("** FINAL OUTPUTS %d **" % batch_id[example_id])
        logging.info(predictions[example_id])
      prediction_tokens = vocab.encode(predictions[example_id])
      results.append((batch_id[example_id], prediction_tokens))

  return results


def main(args):
  logging.basicConfig()
  if args.verbose:
    logging.getLogger().setLevel(logging.INFO)

  logging.info("Loading dataset...")
  seqio_task = f"learning_to_recite:trivia_qa_rc_open_{args.num_shots}shot_fixed_lm_gpt2"

  if args.num_shots != 5:
    raise ValueError("Only 5-shot is supported.")

  evaluator = evaluation.FewShotEvaluator(
      mixture_or_task_name=seqio_task,
      feature_converter=f_converters.DecoderFeatureConverter(pack=False),
      eval_split=args.evaluation_split,
      use_cached=False,
      seed=42,
      num_examples=args.num_examples,
      sequence_length={"inputs": args.input_length, "targets": args.target_length},
      shuffle=False,
  )

  if args.inference_scheme == "recite":
    predict_fn = functools.partial(
        recite_and_answer_fn,
        batch_size=args.batch_size,
        self_consistency_k=args.self_consistency_k,
        verbose=args.verbose
    )
  elif args.inference_scheme == "direct":
    predict_fn = functools.partial(
        direct_answer_fn,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

  else:
    raise ValueError(f"Unknown inference scheme {args.inference_scheme}")

  all_metrics, _, _ = evaluator.evaluate(predict_fn=predict_fn)
  logging.info("Metrics: %s", all_metrics)
  print("Metrics: %s" % all_metrics)


if __name__ == "__main__":
  main(get_args())
