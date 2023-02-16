"""Some seqio preprocessors.
"""

import functools
from typing import Optional, Dict

import seqio
from seqio import utils
import tensorflow.compat.v2 as tf

OutputFeaturesType = seqio.preprocessors.OutputFeaturesType
SequenceLengthType = seqio.preprocessors.SequenceLengthType


def append_eos_after_trim_at_front(
    dataset: tf.data.Dataset,
    output_features: OutputFeaturesType,
    sequence_length: Optional[SequenceLengthType] = None,
) -> tf.data.Dataset:
  """Trims output feature token sequences at front and then appends EOS.

  Args:
    dataset: a tf.data.Dataset of tokenized examples to preprocess.
    output_features: a mapping of output feature names to Feature objects.
    sequence_length: a mapping from output feature names to max lengths. If
      provided, output feature sequences will be trimmed to ensure they are not
      longer than this length once EOS is added.

  Returns:
    a tf.data.Dataset of tokenized examples with EOS added to specified output
    features.
  """
  trim_fn = functools.partial(
      append_eos_after_trim_at_front_impl,
      output_features=output_features,
      sequence_length=sequence_length)
  return utils.map_over_dataset(fn=trim_fn)(dataset)


def append_eos_after_trim_at_front_impl(
    features: Dict[str, tf.Tensor],
    output_features: OutputFeaturesType,
    sequence_length: Optional[SequenceLengthType] = None
) -> Dict[str, tf.Tensor]:
  """Trims the 'input' feature token sequences at front.

  Args:
    features: a dict of tokenized examples to preprocess.
    output_features: a mapping of output feature names to Feature objects.
    sequence_length: a mapping from output feature names to max lengths. If
      provided, output feature sequences will be trimmed to ensure they are not
      longer than this length once EOS is added.

  Returns:
    a tf.data.Dataset of tokenized examples with EOS added to specified output
    features.
  """
  for key, value in features.items():
    if key not in output_features or not output_features[key].add_eos:
      if key == 'inputs':
        if (sequence_length is not None and
            sequence_length.get(key, None) is not None):
          max_length = sequence_length[key]
          features[key] = value[-max_length:]
    else:
      if key == 'inputs':
        eos_id = output_features[key].vocabulary.eos_id
        if (sequence_length is not None and
            sequence_length.get(key, None) is not None):
          max_length = sequence_length[key]
          value = value[-(max_length - 1):]
        features[key] = seqio.preprocessors._append_to_innermost_axis(  # pylint:disable=[protected-access]
            value, eos_id)
  return features


def fewshot_preprocessor(ds,
                         inputs_prefix='',
                         targets_prefix='',
                         example_separator='\n\n',
                         prompt='',
                         train_targets_max_length=None):
  """Create 'inputs' and 'targets' strings for (zero/few)-shot evaluation.

  Inputs and targets will be formatted using the given prefixes along with a
  separator between each pair. The few-shot examples from the train set will
  include both inputs and targets, whereas the eval example (at the end) will
  contain only the input followed by the targets prefix.

  NOTE: The final target prefix will be right-stripped so that the input does
  not end with whitepsace.

  For example, a 2-shot output might look like:
  output: {
    'inputs':
      '0 How many states in the US? X 1 50 X 0 How many cents in a dollar? X '
      '1 100 X 0 Who was in the Beatles? X 1',
    'targets': 'John',
    'answers': ['John', 'Paul', 'George', 'Ringo']
  }

  Args:
    ds: A dictionary of zipped eval and train tf.data.Datasets, each
      preprocessed with at least the fields 'inputs' and 'targets'. Note that
      the train dataset will not exist in the 0-shot case.
    inputs_prefix: Prefix string for inputs.
    targets_prefix: Prefix string for targets.
    example_separator: The string separator to delimit different examples.
    prompt: Optional prefix for the entire few-shot input. Typically
      consists of a natural language description of the task or task
      instructions.
    train_targets_max_length: truncate the targets of prompting examples

  Returns:
    A tf.data.Dataset containing 'inputs', 'targets', and any other features
    from the evaluation dataset.
  """

  @utils.map_over_dataset
  def fewshot_map(ex):
    if 'train' in ex:
      if train_targets_max_length:
        train_examples = tf.reshape(
            tf.stack([
                inputs_prefix + ex['train']['inputs'],
                targets_prefix + tf.strings.strip(
                    ex['train']['targets'])[:train_targets_max_length] +
                example_separator
            ],
                     axis=1), [-1])
      else:
        train_examples = tf.reshape(
            tf.stack(
                [
                    inputs_prefix + ex['train']['inputs'],
                    targets_prefix + ex['train']['targets'] + example_separator
                ],
                axis=1),
            [-1])
      shots = tf.strings.reduce_join(train_examples)
    else:
      shots = ''
    if prompt:
      shots = tf.strings.join([prompt, shots], separator=example_separator)
    new_ex = {
        'inputs':
            shots + inputs_prefix + ex['eval']['inputs'] +
            targets_prefix.rstrip(),
        'targets': ex['eval']['targets'],
    }
    # Pass through other eval features unchanged.
    new_ex.update(
        {k: v for k, v in ex['eval'].items() if k not in ('inputs', 'targets')}
    )
    return new_ex

  ds = fewshot_map(ds)
  if ds.element_spec['inputs'].shape.rank:
    # Unbatch if not a scalar. This is useful for fewshot eval.
    ds = ds.unbatch()
  return ds


def natural_questions_open(
    dataset: tf.data.Dataset,
    prefix: str = 'nq question: ',
    suffix: str = '',
    max_tokens: Optional[int] = None):
  """Convert Natural Questions Open TFDS to examples.

  If there are multiple answers in the input, selects the first one as the
  target.

  The function takes the natural_question_open TFDS dataset and emits examples
  of the form:
  {
    'inputs': 'nq question: What are the names of the Olsen Twins?'
    'targets': 'Mary-Kate and Ashley',
    'answers': ['Mary-Kate and Ashley', 'Ashley and Mary-Kate']
  }

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.
    suffix: str, suffix to append to the targets
    max_tokens: int(optional)

  Returns:
    a tf.data.Dataset
  """

  def nq_map(ex):
    """Map Natural Questions example to text-to-text example."""
    targets = ex['answer'][0]
    if max_tokens:
      targets = tf.strings.substr(targets, 0, max_tokens)
    return {
        'inputs': prefix + ex['question'],
        'targets': targets + suffix,
        'answers': ex['answer'],
    }
  return dataset.map(nq_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def hotpot_qa_open(
    dataset: tf.data.Dataset,
    prefix: str = 'hotpotqa question: ',
    suffix: str = '',
    max_tokens: Optional[int] = None):
  """Convert Huggingface Hotpot QA TFDS to examples.

  If there are multiple answers in the input, selects the first one as the
  target.

  The function takes the natural_question_open TFDS dataset and emits examples
  of the form:
  {
    'inputs': 'nq question: What are the names of the Olsen Twins?'
    'targets': 'Mary-Kate and Ashley',
    'answers': ['Mary-Kate and Ashley', 'Ashley and Mary-Kate']
  }

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.
    suffix: str, suffix to append to the targets
    max_tokens: int(optional)

  Returns:
    a tf.data.Dataset
  """

  def hq_map(ex):
    """Map HQ example to text-to-text example."""
    targets = ex['answer']
    if max_tokens:
      targets = tf.strings.substr(targets, 0, max_tokens)
    return {
        'inputs': prefix + ex['question'],
        'targets': targets + suffix,
        'answers': [ex['answer']],
        'supporting_facts': ex['supporting_facts']['title']
    }
  return dataset.map(hq_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def trivia_qa_open(
    dataset: tf.data.Dataset,
    prefix: str = 'trivia_qa question: ',
    suffix: str = '',
    max_tokens: Optional[int] = None):
  """Convert Natural Questions Open TFDS to examples.

  If there are multiple answers in the input, selects the first one as the
  target.

  The function takes the natural_question_open TFDS dataset and emits examples
  of the form:
  {
    'inputs': 'nq question: What are the names of the Olsen Twins?'
    'targets': 'Mary-Kate and Ashley',
    'answers': ['Mary-Kate and Ashley', 'Ashley and Mary-Kate']
  }

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.
    suffix: str, suffix to append to the targets
    max_tokens: int(optional)

  Returns:
    a tf.data.Dataset
  """

  def nq_map(ex):
    """Map Natural Questions example to text-to-text example."""
    targets = ex['answer']['value']
    if max_tokens:
      targets = tf.strings.substr(targets, 0, max_tokens)
    return {
        'inputs': prefix + ex['question'],
        'targets': targets + suffix,
        'answers': ex['answer']['aliases'],
    }
  return dataset.map(nq_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def truncate_dataset_preprocessor(dataset, take_num=None):
  if take_num:
    dataset = dataset.take(take_num)
  return dataset
