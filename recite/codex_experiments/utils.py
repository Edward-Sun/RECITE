"""Utilities for working with OpenAI GPT APIs.
"""

import json
import logging
import os
import re
import time

from concurrent.futures import ThreadPoolExecutor

import openai
from tasks_and_mixtures.prompts import QA_PROMPT


def openai_complete(
    prompts,
    max_length,
    temperature,
    num_sampling=1,
    sleep_time=3.0, # This is because of the rate limit: 20.000000 / min
  ):
  """OpenAI API call.

  Args:
    prompts: list of prompts
    max_length: max length of the output
    temperature: temperature of the output
    num_sampling: number of sampling
    sleep_time: sleep time to avoid rate limit

  Returns:
    list of responses
  """
  if type(prompts) is str:
    prompts = [prompts]

  def openai_api_call(prompt, api_key, organization):
    response = None
    accumulated_sleep_time = sleep_time
    if len(prompt) > 0:
      while response is None:
        try:
          response = openai.Completion.create(
            model="code-davinci-002",
            prompt=prompt,
            max_tokens=max_length,
            temperature=temperature,
            n=num_sampling,
            api_key=api_key,
            organization=organization,
          )
        except openai.error.RateLimitError as e:
          time.sleep(accumulated_sleep_time)
          accumulated_sleep_time += sleep_time
        except openai.error.APIError as e:
          print(e)
          time.sleep(accumulated_sleep_time)
          accumulated_sleep_time += sleep_time
      return response
    else:
      return None

  api_dicts = []
  multiple_api_key_file = "scripts/openai_keys.json"
  if os.path.exists(multiple_api_key_file):
    with open(multiple_api_key_file, "r") as f:
      for line in f:
        api_dicts.append(json.loads(line))

  if len(api_dicts) == 0:
    api_dicts = [{"api_key": openai.api_key, "organization": openai.organization}]

  targets = []

  logging.info("Using %d API keys" % len(api_dicts))
  with ThreadPoolExecutor(max_workers=len(api_dicts)) as executor:
    futures = []
    for batch_idx, api_dict in enumerate(api_dicts):
      single_process_batch_size = ((len(prompts) - 1) // len(api_dicts)) + 1
      start_idx = single_process_batch_size * batch_idx
      end_idx = single_process_batch_size * (batch_idx + 1)

      if batch_idx == len(api_dicts) - 1:
        single_process_prompts = prompts[start_idx:]
      else:
        single_process_prompts = prompts[start_idx:end_idx]

      futures.append(
          executor.submit(
              openai_api_call,
              single_process_prompts,
              api_dict["api_key"],
              api_dict["organization"],
          ))

    for future in futures:
      response = future.result()
      if response is not None:
        targets.extend([_["text"] for _ in response["choices"]])

  time.sleep(sleep_time)
  return targets


def predict_call(prompt,
                 max_tokens=16,
                 temperature=0.0):
  response = openai_complete(
      prompt,
      max_length=max_tokens,
      temperature=temperature,
  )
  return response


def recite_predict_call(prompt,
                        self_consistency_k,
                        max_tokens=128,
                        temperature=0.7,
                        sentence_splitter="\n\n"):
  results = []
  for _ in range(self_consistency_k):
    response = openai_complete(
        prompt,
        max_length=max_tokens,
        temperature=temperature,
    )

    results.append([_.split(sentence_splitter)[0].strip() for _ in response])
  return results


def cot_predict_call(prompt,
                     self_consistency_k,
                     max_tokens=128,
                     temperature=0.7,
                     sentence_splitter="\n\n"):
  results = []
  response = openai_complete(
      prompt,
      max_length=max_tokens,
      temperature=temperature,
      num_sampling=self_consistency_k,
  )

  for i in range(len(prompt)):
    texts = []
    for j in range(self_consistency_k):
      texts.append(response[i * self_consistency_k + j].split(sentence_splitter)[0].strip())
    results.append(texts)
  return results


def contextual_predict_call(prompts, evidences, max_tokens=16, qa_prompt=QA_PROMPT, temperature=0.0, sentence_splitter="\n\n"):
  results = []
  for evidence in evidences:
    full_qa_prompt = sentence_splitter + qa_prompt + sentence_splitter
    prompts_with_evidence = [
        evi + full_qa_prompt + prompt for evi, prompt in zip(evidence, prompts)]
    response = openai_complete(
        prompts_with_evidence,
        max_length=max_tokens,
        temperature=temperature,
    )
    results.append([_.split("\n")[0].strip() for _ in response])
  results = list(zip(*results))
  return results


def hotpotqa_answer_postprocess(cot_answer, qa_prompt=False):
  """Post-process hq answer before voting."""
  if qa_prompt:
    cot_answer = cot_answer.split('\n\n')[0]
    cot_answer = cot_answer.split(' [eot]')[0]
    cot_answer = cot_answer.split(' [eod]')[0]
    cot_answer = cot_answer + '\n'
  else:
    cot_answer = cot_answer.split('\n')[0]
    cot_answer = cot_answer.split(' [eot]')[0]
    cot_answer = cot_answer + '\n'
  answers = re.findall(r'The answer is (.*?)\.\n', cot_answer)
  if answers:
    return answers[0]
  else:
    return ''


def get_majority_voting(multiple_outputs, post_process_fn=None):
  """Get majority voting from list of strings."""
  majority_voting = {}
  for ans in multiple_outputs:
    if post_process_fn is not None:
      ans = post_process_fn(ans)
    if ans not in majority_voting:
      majority_voting[ans] = 1
    else:
      majority_voting[ans] += 1

  best_ans = list(majority_voting.keys())[0]

  for ans in majority_voting:
    if not ans:
      continue
    if (majority_voting[ans] > majority_voting[best_ans]) or not best_ans:
      best_ans = ans
    if majority_voting[ans] == majority_voting[best_ans] and len(ans) < len(best_ans):
      best_ans = ans

  return best_ans
