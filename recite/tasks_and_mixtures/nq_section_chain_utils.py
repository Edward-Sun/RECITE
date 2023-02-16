"""Utility functions for section hints.
"""

import functools
import random
import re
from typing import Optional

import html2text
import numpy as np
import tensorflow as tf


def html2text_tf(text: tf.Tensor):  # pylint: disable=[missing-function-docstring]
  pure_text = text.numpy().decode("utf-8")
  parser = html2text.HTML2Text()
  parser.body_width = 0
  parser.ignore_links = True
  parser.ignore_images = True
  parser.ignore_emphasis = True
  parser.wrap_links = False
  parser.wrap_list_items = False
  parser.ul_item_mark = ""
  parser.emphasis_mark = ""
  parser.strong_mark = ""
  pure_text = pure_text.replace("<BR />", " ; ")
  pure_text = parser.handle(pure_text).strip()
  pure_text = re.sub(r"\[.*?\]", "", pure_text)
  return tf.strings.join([pure_text])


def get_section_chain_paragraphs_and_tables(
    document: tf.Tensor,
    top_k_bytes: Optional[int] = None,
    top_k_passages: Optional[int] = None,
    minimum_passage_length: int = 128,
    with_date: bool = False,
    balance_short_description_ratio: Optional[int] = None):
  """Extract pragraphs and tables from Wikipedia html page with section chain.

  Args:
      document: an NQ document example
      top_k_bytes: int, only extracting passages in top-k bytes
      top_k_passages: int, only extracting first-k passages in each section
      minimum_passage_length: filter out too-short passages
      with_date: append date to section chain
      balance_short_description_ratio: balance the portion of short description

  Returns:
      two string vectors for paired section chain & passages
  """
  if top_k_bytes and top_k_passages:
    raise ValueError("top_k_tokens and top_k_passages can be set togather")

  heading_tags = [b"<H1>", b"<H2>", b"<H3>", b"<H4>", b"<H5>", b"<H6>"]
  heading_end_tags = [
      b"</H1>", b"</H2>", b"</H3>", b"</H4>", b"</H5>", b"</H6>"
  ]
  list_ptags = [b"<UL", b"<OL", b"<DL"]
  list_end_tags = [b"</UL>", b"</OL>", b"</DL>"]

  document = document.numpy()
  document = b"<H1" + document.split(b"<H1", 1)[1]
  document = re.sub(r"<(H[0-9])\b.*?>", r"<\1>", document.decode("utf-8"))

  wiki_date = re.split("This page was last edited on ", document)
  if len(wiki_date) > 1:
    wiki_date = wiki_date[1].split(",")[0]
  else:
    wiki_date = "20 December 2018"
  wiki_date = wiki_date.encode("utf-8")
  wiki_date = b" (edited on %s)" % wiki_date

  document = re.split(r"(<[^>]+>)", document)
  document = [_.encode("utf-8") for _ in document if _.strip()]

  current_section_titles = []
  title_results = [b""]
  content_results = [b""]
  sub_results = []
  start_recording_headline = False
  current_category = None
  paragraph_category_number = 0
  table_category_number = 0
  list_category_number = 0
  accumulated_string_length = 0
  element_hierachical_level = 0
  for idx in range(0, len(document)):
    sub_results.append(document[idx])
    accumulated_string_length += len(document[idx])
    if document[idx] in heading_end_tags:
      start_recording_headline = False
      current_section_titles[-1] = b"".join(current_section_titles[-1])
      reuse_level = {
          b"</H1>": 1,
          b"</H2>": 2,
          b"</H3>": 3,
          b"</H4>": 4,
          b"</H5>": 5,
          b"</H6>": 6,
      }[document[idx]]
      current_section_titles = (
          current_section_titles[:reuse_level - 1] +
          current_section_titles[-1:])
      sub_results = []
      if len(current_section_titles) == 2:
        # pylint:disable=[g-complex-comprehension]
        if any([
            stop_words in current_section_titles[-1] for stop_words in [
                b"See also", b"References", b"External links", b"Related lists",
                b"Navigation menu"
            ]
        ]):
          break
        # pylint:enable=[g-complex-comprehension]

    if start_recording_headline:
      current_section_titles[-1].append(document[idx])

    if ((document[idx] in [b"<P>"] + heading_tags) or
        document[idx].startswith(b"<TABLE")) and element_hierachical_level == 0:
      if len(sub_results) > 1:
        sub_results_start_idx = (
            accumulated_string_length - len(b"".join(sub_results)))
        if len(current_section_titles) >= 1:
          sub_results = b"".join(sub_results[:-1])
          new_section_titles = current_section_titles + [None]
          del new_section_titles[-1]
          if len(new_section_titles) == 1:
            new_section_titles = new_section_titles + [b"Short description"]
          if current_category == b"<P>":
            new_section_titles.append(b"Paragraph #%d" %
                                      paragraph_category_number)
          elif current_category == b"<TABLE>":
            new_section_titles.append(b"Table #%d" % table_category_number)
          else:
            new_section_titles.append(b"List #%d" % list_category_number)

          if sub_results.startswith(b"<P></P>"):
            paragraph_category_number -= 1
          elif sub_results.startswith(b'<TABLE class="plainlinks'):
            table_category_number -= 1
          elif (len(new_section_titles) == 3 and
                new_section_titles[-2] == b"Contents"):
            pass
          elif new_section_titles[-1] == b"List #0":
            pass
          elif top_k_bytes and sub_results_start_idx > top_k_bytes:
            pass
          elif top_k_passages and (
              (current_category == b"<P>" and
               paragraph_category_number > top_k_passages) or
              (current_category == b"<TABLE>" and
               table_category_number > top_k_passages)):
            pass
          elif not (new_section_titles[-2] == b"Short description" and
                    current_category is None):
            if len(sub_results) > minimum_passage_length:
              new_section_titles = b" ; ".join(new_section_titles)
              if with_date:
                new_section_titles = new_section_titles + wiki_date
              title_results.append(new_section_titles)
              content_results.append(sub_results)
      sub_results = [document[idx]]
      current_category = None

    if ((document[idx] in [b"<P>"]) or any(
        [document[idx].startswith(tag) for tag in (list_ptags + [b"<TABLE"])])):
      element_hierachical_level += 1
      if current_category is None:
        if document[idx] in [b"<P>"]:
          current_category = document[idx]
        elif document[idx].startswith(b"<TABLE"):
          current_category = b"<TABLE>"
        else:
          current_category = b"<LIST>"

        if document[idx] == b"<P>":
          paragraph_category_number += 1
        if document[idx].startswith(b"<TABLE"):
          table_category_number += 1
        if any([document[idx].startswith(tag) for tag in list_ptags]):
          if element_hierachical_level == 1:
            list_category_number += 1
      sub_results = sub_results[-1:]

    if document[idx] in [b"</P>", b"</TABLE>"] + list_end_tags:
      element_hierachical_level -= 1

    if document[idx] in heading_tags:
      start_recording_headline = True
      current_section_titles.append([])
      paragraph_category_number = 0
      table_category_number = 0
      list_category_number = 0

  if balance_short_description_ratio is None:
    return tf.constant(title_results), tf.constant(content_results)
  else:
    s_title_results = []
    s_content_results = []
    ns_title_results = []
    ns_content_results = []

    for title_result, content_result in zip(title_results, content_results):
      if len(content_result) > 1:
        if b"Short description" in title_result:
          s_title_results.append(title_result)
          s_content_results.append(content_result)
        else:
          ns_title_results.append(title_result)
          ns_content_results.append(content_result)

    ns_results = list(zip(ns_title_results, ns_content_results))
    random.shuffle(ns_results)
    if s_title_results:
      ns_results = ns_results[:(len(s_title_results) *
                                balance_short_description_ratio)]

    ns_title_results = [_[0] for _ in ns_results]
    ns_content_results = [_[1] for _ in ns_results]

    if not s_title_results + ns_title_results:
      s_title_results.append("")
      s_content_results.append("")

    return (tf.constant(s_title_results + ns_title_results),
            tf.constant(s_content_results + ns_content_results))


def get_section_chain_from_long_answer(document: tf.Tensor,
                                       long_answer_start_byte: tf.Tensor):
  """Get long answer annotation with section chain.

  Args:
      document: an NQ document example
      long_answer_start_byte: integer, start byte of long answer

  Returns:
      two strings for section chain & page date
  """
  heading_tags = [b"<H1>", b"<H2>", b"<H3>", b"<H4>", b"<H5>", b"<H6>"]
  heading_end_tags = [
      b"</H1>", b"</H2>", b"</H3>", b"</H4>", b"</H5>", b"</H6>"
  ]
  list_ptags = [b"<UL", b"<OL", b"<DL"]
  list_end_tags = [b"</UL>", b"</OL>", b"</DL>"]

  document = document.numpy()
  h1_prefix, document = document.split(b"<H1", 1)
  document = b"<H1" + document
  document = document.decode("utf-8")

  wiki_date = re.split("This page was last edited on ", document)
  if len(wiki_date) > 1:
    wiki_date = wiki_date[1].split(",")[0]
  else:
    wiki_date = "20 December 2018"
  wiki_date = wiki_date.encode("utf-8")

  document = re.split(r"(<[^>]+>)", document)
  document = [_.encode("utf-8") for _ in document]
  long_answer_start_byte = int(long_answer_start_byte)
  if long_answer_start_byte == -1:
    return b"", wiki_date

  current_section_titles = []
  sub_results = []
  start_recording_headline = False
  current_category = None
  paragraph_category_number = 0
  table_category_number = 0
  list_category_number = 0
  accumulated_string_length = len(h1_prefix)
  element_hierachical_level = 0
  for idx in range(0, len(document)):
    sub_results.append(document[idx])
    accumulated_string_length += len(document[idx])

    if accumulated_string_length > long_answer_start_byte:
      if document[idx] == b"<P>":
        ground_truth_category = b"<P>"
        paragraph_category_number += 1
      elif (document[idx].startswith(b"<TABLE") or
            document[idx].startswith(b"<TR")):
        ground_truth_category = b"<TABLE>"
        table_category_number += 1
      else:
        ground_truth_category = b"<LIST>"
        list_category_number += 1

      new_section_titles = current_section_titles + [None]
      del new_section_titles[-1]
      if len(new_section_titles) == 1:
        new_section_titles = new_section_titles + [b"Short description"]
      if ground_truth_category == b"<P>":
        new_section_titles.append(b"Paragraph #%d" % paragraph_category_number)
      elif ground_truth_category == b"<TABLE>":
        new_section_titles.append(b"Table #%d" % table_category_number)
      else:
        new_section_titles.append(b"List #%d" % list_category_number)
      return b" ; ".join(new_section_titles), wiki_date

    if document[idx] in heading_end_tags:
      start_recording_headline = False
      current_section_titles[-1] = b"".join(current_section_titles[-1])
      reuse_level = {
          b"</H1>": 1,
          b"</H2>": 2,
          b"</H3>": 3,
          b"</H4>": 4,
          b"</H5>": 5,
          b"</H6>": 6,
      }[document[idx]]
      current_section_titles = (
          current_section_titles[:reuse_level - 1] +
          current_section_titles[-1:])
      sub_results = []

    if start_recording_headline:
      current_section_titles[-1].append(document[idx])

    if ((document[idx] in [b"<P>"] + heading_tags) or
        document[idx].startswith(b"<TABLE")) and element_hierachical_level == 0:
      if len(sub_results) > 1:
        sub_results = b"".join(sub_results[:-1])
        if sub_results.startswith(b"<P></P>"):
          paragraph_category_number -= 1
        elif sub_results.startswith(b'<TABLE class="plainlinks'):
          table_category_number -= 1
      sub_results = [document[idx]]
      current_category = None

    if ((document[idx] in [b"<P>"]) or any(
        [document[idx].startswith(tag) for tag in (list_ptags + [b"<TABLE"])])):
      element_hierachical_level += 1
      if current_category is None:
        if document[idx] in [b"<P>"]:
          current_category = document[idx]
        elif document[idx].startswith(b"<TABLE"):
          current_category = b"<TABLE>"
        else:
          current_category = b"<LIST>"

        if document[idx] == b"<P>":
          paragraph_category_number += 1
        if document[idx].startswith(b"<TABLE"):
          table_category_number += 1
        if any([document[idx].startswith(tag) for tag in list_ptags]):
          if element_hierachical_level == 1:
            list_category_number += 1
      sub_results = sub_results[-1:]

    if document[idx] in [b"</P>", b"</TABLE>"] + list_end_tags:
      element_hierachical_level -= 1

    if any([
        document[idx].startswith(htag)
        for htag in [b"<H1", b"<H2", b"<H3", b"<H4", b"<H5", b"<H6"]
    ]):
      start_recording_headline = True
      current_section_titles.append([])
      paragraph_category_number = 0
      table_category_number = 0
      list_category_number = 0
  return b"Unknown Wikipedia section chain", wiki_date


def batched_clean_nq_map(section_titles,
                         section_contents,
                         sample_k_passage=7,
                         max_byte_length=512,
                         min_byte_length=32):
  """Batch-clean the section titles and contents from html form."""

  def _clean_nq_map(section_titles, section_contents):
    passage_num = section_titles.shape[0]

    passage_choice = np.random.choice(passage_num, passage_num, replace=False)

    sampled_section_titles = []
    sampled_section_contents = []

    idx = 0
    while len(sampled_section_titles) < sample_k_passage and idx < passage_num:
      choice_idx = passage_choice[idx]
      section_title = section_titles[choice_idx]
      section_title = html2text_tf(section_title)
      section_title = tf.strings.regex_replace(section_title, " ; ", " --- ")

      section_content = section_contents[choice_idx]
      section_content = html2text_tf(section_content)
      section_content = tf.strings.regex_replace(section_content, "\n[ ]+",
                                                 "\n")
      section_content = tf.strings.regex_replace(section_content, "\n[\n]+",
                                                 "\n\n")

      if tf.strings.length(section_content).numpy() > min_byte_length:
        sampled_section_titles.append(section_title.numpy()[:max_byte_length])
        sampled_section_contents.append(
            section_content.numpy()[:max_byte_length])
      idx += 1

    if not sampled_section_titles:
      sampled_section_titles.append("")
      sampled_section_contents.append("")
    return sampled_section_titles, sampled_section_contents

  section_titles, section_contents = tf.py_function(
      func=_clean_nq_map,
      inp=[section_titles, section_contents],
      Tout=[tf.string, tf.string])
  section_titles = tf.reshape(section_titles, (-1,))
  section_contents = tf.reshape(section_contents, (-1,))

  return section_titles, section_contents


def natural_questions_with_section_chain(dataset: tf.data.Dataset,
                                         prefix: str = "",
                                         drop_yes_no: bool = False,
                                         max_tokens: Optional[int] = None,
                                         max_answers: Optional[int] = None,
                                         max_byte_length: Optional[int] = None):
  """Convert Natural Questions TFDS to question => section chain + passage.

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.
    drop_yes_no: bool, whether to drop yes/no answers, keeping only short
      answers.
    max_tokens: (Optional) int, the maximum number of tokens (as specified by
      NQ) beyond which a short answer is dropped. None are dropped if set to
      `None`.
    max_answers: (Optional) int, the maximum number of answers to include in the
      targets. Will be selected deterministically from the beginning of the
      list. All answers are included if set to `None`.
    max_byte_length: maximum byte length of each passage

  Returns:
    a tf.data.Dataset
  """

  def nq_map(ex):
    """Map Natural Questions example to text-to-text example."""
    inputs = prefix + ex["question"]["text"]

    annotations = ex["annotations"]
    context = ex["document"]["html"]

    non_blank_index = tf.argmax(
        annotations["long_answer"]["end_byte"] -
        annotations["long_answer"]["start_byte"],
        axis=0)
    long_answer_start_byte = annotations["long_answer"]["start_byte"][
        non_blank_index]
    long_answer_end_byte = annotations["long_answer"]["end_byte"][
        non_blank_index]

    section_title, page_date = tf.py_function(
        func=get_section_chain_from_long_answer,
        inp=[context, long_answer_start_byte],
        Tout=[tf.string, tf.string])
    section_title = tf.py_function(
        func=html2text_tf, inp=[section_title], Tout=tf.string)
    section_title = tf.strings.regex_replace(section_title, " ; ", " --- ")

    long_answer = tf.strings.substr(
        context, long_answer_start_byte,
        (long_answer_end_byte - long_answer_start_byte))
    long_answer = tf.py_function(
        func=html2text_tf, inp=[long_answer], Tout=tf.string)
    long_answer = tf.strings.regex_replace(long_answer, "\n[ ]+", "\n")
    long_answer = tf.strings.regex_replace(long_answer, "\n[\n]+", "\n\n")

    section_title = tf.reshape(section_title, shape=())
    long_answer = tf.reshape(long_answer, shape=())

    long_answer = tf.where(
        annotations["long_answer"]["start_byte"][non_blank_index] >= 0,
        long_answer, "")

    yes_no_labels = annotations["yes_no_answer"]
    if drop_yes_no:
      yes_no_labels = -1 * tf.ones_like(yes_no_labels)
    yes_no_answers = tf.boolean_mask(yes_no_labels, yes_no_labels > -1)
    yes_no_answers = tf.where(tf.equal(yes_no_answers, 1), "yes", "no")

    short_answers = annotations["short_answers"]["text"].flat_values
    short_answer_starts = annotations["short_answers"]["text"].row_starts()
    if max_tokens:
      start_tokens = annotations["short_answers"]["start_token"]
      end_tokens = annotations["short_answers"]["end_token"]
      dropped_answers = end_tokens - start_tokens > max_tokens
      short_answers = tf.boolean_mask(
          short_answers, tf.math.logical_not(dropped_answers.values))
      # Subtract dropped answers from row starts.
      row_drop_count = tf.math.reduce_sum(
          tf.cast(dropped_answers, tf.int64), axis=1)
      short_answer_starts -= tf.concat(
          [[0], tf.math.cumsum(row_drop_count[:-1])], axis=0)

    answers = tf.concat([yes_no_answers, short_answers], axis=0)
    if max_answers:
      answers = answers[:max_answers]
    targets = tf.strings.reduce_join(answers, separator=" <extra_id_1> ")

    if max_byte_length:
      long_answer = tf.strings.substr(long_answer, 0, max_byte_length)

    return {
        "section_title": section_title,
        "title": ex["document"]["title"],
        "inputs": inputs,
        "targets": targets,
        "long_answer": long_answer,
        "answers": short_answers,
    }

  dataset = dataset.map(
      nq_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.filter(lambda ex: tf.strings.length(ex["targets"]) > 0)
  return dataset.filter(lambda ex: tf.strings.length(ex["long_answer"]) > 0)

