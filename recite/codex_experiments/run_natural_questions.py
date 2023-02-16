"""Experiments for Natural Questions.
"""

import argparse
import json
import logging
import os
import functools

import requests
import seqio.feature_converters as f_converters
import tensorflow as tf
from requests.structures import CaseInsensitiveDict

import codex_experiments.evaluation as evaluation
from codex_experiments.utils import predict_call, recite_predict_call, contextual_predict_call
from codex_experiments.utils import get_majority_voting
import tasks_and_mixtures.few_shot_nq_tasks as few_shot_nq_tasks  # pylint: disable=unused-import
from tasks_and_mixtures.prompts import QA_PROMPT, SHORT_ANSWER_PROMPT

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
  parser.add_argument("--retrieve_url", type=str, default="http://localhost:5000/retrieve")

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


def nq_input_processing_fn(inputs,
                           vocab,
                           exemplar_splitter="\n\n\n",
                           sentence_splitter="\n\n"):
  decoder_input_tokens, decoder_causal_attention = inputs
  decoder_input_tokens = decoder_input_tokens[decoder_causal_attention]
  decoder_input_tokens = decoder_input_tokens[:-1]
  decoder_inputs = vocab.decode(decoder_input_tokens)

  exemplars = decoder_inputs.split(exemplar_splitter)
  if len(exemplars[0].split(sentence_splitter)) < 4:
    exemplars = exemplars[1:]

  recitation_inputs = [sentence_splitter.join(_.split(sentence_splitter)[:3]) for _ in exemplars]
  recitation_inputs = exemplar_splitter.join(recitation_inputs)

  short_qa_inputs = [
      _.split(sentence_splitter)[0] + sentence_splitter + _.split(sentence_splitter)[-1] for _ in exemplars]
  short_qa_inputs = exemplar_splitter.join(short_qa_inputs)
  short_qa_inputs = sentence_splitter + short_qa_inputs
  short_qa_inputs = short_qa_inputs.replace(
      f"{sentence_splitter}Question: ", f"{sentence_splitter}Q: ")
  short_qa_inputs = short_qa_inputs.replace(
      f"{sentence_splitter}{SHORT_ANSWER_PROMPT}", f"{sentence_splitter}A: ")

  # Process the test case for the short answer prompt
  short_qa_inputs = short_qa_inputs.replace(
      f"{sentence_splitter}Answer:", f"{sentence_splitter}A:")
  short_qa_inputs = short_qa_inputs[len(sentence_splitter):]

  return recitation_inputs, short_qa_inputs


def nq_target_processing_fn(inputs, vocab):
  decoder_input_tokens, decoder_loss_weights = inputs
  decoder_input_tokens = decoder_input_tokens[decoder_loss_weights]
  decoder_inputs = vocab.decode(decoder_input_tokens)
  return decoder_inputs


def retrieve_bm25(inputs, retrieval_url):
  data = {
      "inputs": [inputs],
      "n": 1,
  }
  data = json.dumps(data)

  headers = CaseInsensitiveDict()
  headers["Content-Type"] = "application/json"
  generate_resp = requests.post(retrieval_url, headers=headers, data=data)
  targets = generate_resp.json()['targets']

  response = targets[0]
  return response


def direct_answer_fn(dataset, task,
                     batch_size,
                     exemplar_splitter="\n\n\n",
                     sentence_splitter="\n\n",
                     use_ground_truth_context=False,
                     use_bm25=False,
                     retrieve_url=None,
                     verbose=False):
  """Answer a question directly or using contexts.

  Args:
    dataset: A tf.data.Dataset.
    task: A seqio.Task.
    batch_size: Batch size for evaluation.
    exemplar_splitter: The string used to split the exemplars.
    sentence_splitter: The string used to split the sentences within an exemplar.
    use_ground_truth_context: Whether to use the ground truth context.
    use_bm25: Whether to use BM25 to retrieve the context.
    retrieve_url: The URL to use for retrieving the context.
    verbose: Whether to print the input and output.

  Returns:
    A list of tokenized answers.
  """

  results = []
  vocab = task.output_features["targets"].vocabulary
  dataset = dataset.batch(batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE)

  get_decoder_inputs = functools.partial(
      nq_input_processing_fn,
      vocab=vocab,
      exemplar_splitter=exemplar_splitter,
      sentence_splitter=sentence_splitter)

  get_decoder_targets = functools.partial(
      nq_target_processing_fn,
      vocab=vocab)

  for batch_id, ex in dataset.as_numpy_iterator():
    all_tokens = ex["decoder_target_tokens"]
    causal_attention = ex["decoder_causal_attention"] > 0
    task_inputs = list(map(get_decoder_inputs, zip(all_tokens, causal_attention)))
    qa_inputs = [_[1] for _ in task_inputs]

    if use_ground_truth_context and use_bm25:
      raise ValueError("Cannot use both ground truth context and BM25.")

    full_qa_prompt = sentence_splitter + QA_PROMPT + exemplar_splitter

    if use_ground_truth_context:
      loss_weights = ex["decoder_loss_weights"] > 0
      task_targets = list(map(get_decoder_targets, zip(all_tokens, loss_weights)))
      contexts = [dec_tar.split(sentence_splitter)[0] for dec_tar in task_targets]

      qa_inputs = [context + full_qa_prompt+ dec_in for dec_in, context in zip(qa_inputs, contexts)]

    if use_bm25:
      bm25_queries = [_.split(sentence_splitter)[-2] for _ in qa_inputs]
      retrieve_bm25_with_url = functools.partial(retrieve_bm25, retrieval_url=retrieve_url)
      contexts = list(map(retrieve_bm25_with_url, bm25_queries))
      qa_inputs = [context + full_qa_prompt+ dec_in for dec_in, context in zip(qa_inputs, contexts)]

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
      nq_input_processing_fn,
      vocab=vocab,
      exemplar_splitter=exemplar_splitter,
      sentence_splitter=sentence_splitter)

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
        logging.info("** QA INPUTS %d **" % batch_id[example_id])
        logging.info(qa_inputs[example_id])

    qa_predictions = contextual_predict_call(qa_inputs, recite_predictions)
    predictions = list(map(get_majority_voting, qa_predictions))

    for example_id in range(batch_id.shape[0]):
      if verbose:
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
  seqio_task = f"learning_to_recite:natural_questions_open_long_answer_{args.num_shots}shot_lm_gpt2"

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

  elif args.inference_scheme == "with_ground_truth_context":
    predict_fn = functools.partial(
        direct_answer_fn,
        batch_size=args.batch_size,
        verbose=args.verbose,
        use_ground_truth_context=True,
    )

  elif args.inference_scheme == "with_bm25_context":
    predict_fn = functools.partial(
        direct_answer_fn,
        batch_size=args.batch_size,
        verbose=args.verbose,
        use_bm25=True,
        retrieve_url=args.retrieve_url,
    )

  else:
    raise ValueError(f"Unknown inference scheme {args.inference_scheme}")

  all_metrics, _, _ = evaluator.evaluate(predict_fn=predict_fn)
  logging.info("Metrics: %s", all_metrics)
  print("Metrics: %s" % all_metrics)


if __name__ == "__main__":
  main(get_args())
