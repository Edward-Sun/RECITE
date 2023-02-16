"""Seqio vocabularies for gpt2 and gpt3"""
from typing import Optional

import numpy as np
import seqio.vocabularies as vocabularies
import tensorflow as tf
from transformers import GPT2Tokenizer


Vocabulary = vocabularies.Vocabulary


class GPT2Vocabulary(Vocabulary):
  """Wrapper for GPT2 tokenizer.
  """

  def __init__(self):
    """Vocabulary constructor."""
    # We won't pass in extra_ids for Bert vocabulary.
    self.hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    super().__init__()

  @property
  def sos_id(self) -> Optional[int]:
    return self.hf_tokenizer.bos_token_id

  @property
  def eos_id(self) -> Optional[int]:
    return self.hf_tokenizer.eos_token_id

  @property
  def unk_id(self) -> Optional[int]:
    return self.hf_tokenizer.unk_token_id

  @property
  def pad_id(self) -> Optional[int]:
    return self.hf_tokenizer.pad_token_id

  @property
  def _base_vocab_size(self):
    """Returns the vocabulary size."""
    return self.hf_tokenizer.vocab_size

  @property
  def vocab_size(self):
    return self._base_vocab_size

  def _encode(self, s):
    """Encode a python string as a list of integers.
    Args:
      s: a string
    Returns:
      a list of integers (not terminated by EOS)
    """
    return self.hf_tokenizer.encode(s)

  def _decode(self, ids):
    """Decode a list of integers to a python string.
    Args:
      ids: a list of integers (not terminated by EOS)
    Returns:
      a string
    """
    return self.hf_tokenizer.decode(ids)

  def _encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.
    This will be necessary for on-the-fly tokenization.
    Args:
      s: a tf.Scalar with dtype tf.string
    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """

    def tf_encode(inputs):
      return np.array(self._encode(inputs.numpy().decode("utf-8", errors="ignore")))

    ids = tf.py_function(
        func=tf_encode,
        inp=[s],
        Tout=tf.int32,
    )
    return tf.reshape(ids, (-1,))

  def _decode_tf(self, ids):
    """Decode in TensorFlow.
    Args:
      ids: a 1d tf.Tensor with dtype tf.int32
    Returns:
      a tf Scalar with dtype tf.string
    """
    return self.hf_tokenizer.decode(ids)

  def __eq__(self, other):
    if not isinstance(other, GPT2Vocabulary):
      return False
    return (self.vocab_size == other.vocab_size and
            self.eos_id == other.eos_id)
