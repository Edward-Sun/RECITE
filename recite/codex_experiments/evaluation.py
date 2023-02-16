"""Few-shot evaluator for prediction service.
"""

import concurrent
import functools
import inspect
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Type

import seqio
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tqdm
import typing_extensions
from absl import logging
from seqio import dataset_providers
from seqio import feature_converters
from seqio import loggers as loggers_lib

Task = dataset_providers.Task
FeatureConverter = feature_converters.FeatureConverter


class FewShotPredictFnCallable(typing_extensions.Protocol):

  def __call__(
      self, dataset: tf.data.Dataset,
      task: dataset_providers.Task,
      model_feature_shapes: Optional[Mapping[str, int]]
  ) -> seqio.evaluation.PredictFnReturnType:
    ...


def get_all_tasks(tasks: Sequence[Task], split: str) -> Sequence[Task]:
  """Get tasks that have the specified split and a metric function."""

  valid_tasks = []

  for task in tasks:
    if split not in task.splits:
      logging.info("Task %s has no '%s' split; skipping eval.", task.name,
                   split)
      continue
    metric_types = []
    if task.predict_metric_fns:
      metric_types.append("predict")
    if task.predict_with_aux_metric_fns:
      metric_types.append("predict_with_aux")
    if task.score_metric_fns:
      metric_types.append("score")
    logging.info("Adding task '%s' with %s metric_fn(s).", task.name,
                 " and ".join(metric_types))
    valid_tasks.append(task)

  return valid_tasks


class FewShotEvaluator(seqio.evaluation.Evaluator):
  """A class to encapsulate all few-shot eval-related information.

  Attributes:
    eval_tasks: a mapping from a mixture or a task name to seqio.Task object(s).
    cached_model_datasets: cached evaluation datasets with model features.
    cached_task_datasets: cached evaluation datasets with task features.
    cached_targets: cached evaluation targets.
    model_feature_shapes: mapping from model feature to its shape in the
      `cached_model_datasets`.
    loggers: a sequence of subclasses of `Logger`.
  """

  def __init__(self,
               mixture_or_task_name: str,
               feature_converter: FeatureConverter,
               eval_split: str = "validation",
               use_cached: bool = False,
               seed: Optional[int] = 42,
               sequence_length: Optional[Mapping[str, int]] = None,
               num_examples: Optional[int] = None,
               shuffle: bool = False,
               logger_cls: Sequence[Type[loggers_lib.Logger]] = (),
               log_dir: Optional[str] = None,
               use_memory_cache: bool = True,
               target_field_name: str = "targets"):
    """Evaluator constructor.

    Args:
      mixture_or_task_name: a registered task or mixture name.
      feature_converter: a feature converter object to use to convert the task
        features to model features. Must be a subclass of
        seqio.FeatureConverter.
      eval_split: evaluation split. Typically "validation" or "test".
      use_cached: whether to use the cached dataset instead of processing it on
        the fly.
      seed: random seed used for dataset shuffle and preprocessing. This is
        usually not needed since eval datasets aren't shuffled and shouldn't use
        stochastic operations. It is only useful for in certain data sources
        such as `FewshotDataSource` where the training examples are randomly
        selected during evaluation.
      sequence_length: an optional length specification. If specified, these
        will be the hard-limit on the evaluation data used for prediction. If
        none of the preprocessors depend on the sequence length, it can be left
        unspecified and the maximum length for each feature will be used. These
        lengths are computed while caching the datasets.
      num_examples: an optional maximum number of examples to take from the
        beginning of each Task dataset for evaluation.
      shuffle: whether to shuffle the Task datasets. Only useful when
        `num_examples` is also set in order to get a semi-random subsample of
        the examples. Note that the shuffle will only be applied once during
        initialization (using `seed`) and the same subsample will be used on
        call to `evaluate`.
      logger_cls: a set of subclasses of `Logger` to write results with.
      log_dir: the directory to log outputs to. Required if `logger_cls` is
        non-empty.
      use_memory_cache: whether to use tf.data.Dataset#cache. may cause memory
        issues for large datasets.
      target_field_name: Field name of the target in the input dataset examples.

    Raises:
      ValueError if `sequence_length` is None but a preprocessor depends on its
      value.
    """
    logging.info("Initializing Evaluator for '%s'", mixture_or_task_name)
    eval_tasks = dataset_providers.get_subtasks(
        dataset_providers.get_mixture_or_task(mixture_or_task_name))
    self._eval_tasks = seqio.evaluation.get_valid_eval_tasks(
        eval_tasks, eval_split)
    self._all_tasks = get_all_tasks(eval_tasks, eval_split)

    self._metrics_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1)
    self._metrics_future = None
    self._target_field_name = target_field_name

    if not self._all_tasks:
      logging.warning(
          "No eval task with valid split and metric fn found. Skipping eval.")
      return

    # Determine if sequence_length arg is required. This occurs when any of the
    # task preprocessors have a `sequence_length` arg with no default value.
    sequence_length_required = False
    for task in self._all_tasks:
      for prep in task.preprocessors:
        prep_params = inspect.signature(prep).parameters
        if ("sequence_length" in prep_params and
            prep_params["sequence_length"].default == inspect.Parameter.empty):
          if sequence_length is None:
            if isinstance(prep, functools.partial):
              prep_name = prep.func.__name__
            else:
              prep_name = prep.__name__
            raise ValueError(
                f"Preprocessor '{prep_name}' in task '{task.name}' has a "
                "`sequence_length` argument, making it incompatible with "
                "automatic sequence length detection. Pass a valid "
                "`sequence_length` to `Evaluator` and try again.")
          sequence_length_required = True
          break

    def dataset_fn(task: Task) -> tf.data.Dataset:
      try:
        return task.get_dataset(
            sequence_length=sequence_length,
            split=eval_split,
            shuffle=shuffle,
            num_epochs=1,
            seed=seed,
            use_cached=use_cached)
      except AssertionError:
        print("**Cannot find cached dataset, building from scratch**")
        return task.get_dataset(
            sequence_length=sequence_length,
            split=eval_split,
            shuffle=False,
            shuffle_buffer_size=1,
            num_epochs=1,
            seed=seed,
            use_cached=False)

    # `task_datasets` have the output features from seqio.Task.get_dataset.
    # These features will be converted to "model features" by the feature
    # converter before being cached.
    sequence_dims = {
        k: v.sequence_dim for k, v in feature_converter.TASK_FEATURES.items()
    }

    if self._eval_tasks:
      cached_targets, cached_task_datasets, max_lengths = (
          get_targets_and_examples(
              tasks=self._eval_tasks,
              dataset_fn=dataset_fn,
              sequence_dims=sequence_dims,
              num_examples=num_examples,
              use_memory_cache=use_memory_cache,
              target_field_name=self._target_field_name))
    else:
      cached_targets = {}
      cached_task_datasets = {}
      max_lengths = {
          k: 777 for k in self._all_tasks[0].output_features.keys()}

    for task in self._all_tasks:
      if task not in self._eval_tasks:
        cached_targets = dict(cached_targets)
        cached_task_datasets = dict(cached_task_datasets)
        cached_targets[task.name] = []
        cached_task_datasets[task.name] = dataset_fn(task)

    if sequence_length is None:
      logging.info("Setting sequence lengths to %s", max_lengths)
      sequence_length = max_lengths
    else:
      log_long_warning = False
      log_same_warning = False

      sequence_length = {
          k: sequence_length.get(k, max_lengths[k]) for k in max_lengths
      }

      assert set(sequence_length.keys()) == set(max_lengths.keys()), (
          "sequence_length=%s limits must match the detected max_lengths=%s" %
          (sequence_length.keys(), max_lengths.keys()))

      for k, l in sequence_length.items():
        if l is None:
          continue
        if isinstance(l, (tuple, list)):
          logging.warning(
              "Automatic length checking is not supported when lengths are"
              "specified with a tuple for feature %s = %s. Please make "
              "sure your max lengths are not removing parts of your inputs.", k,
              l)
        elif l > max_lengths[k]:
          log_long_warning = True
        elif not sequence_length_required and l == max_lengths[k]:
          log_same_warning = True

      if log_long_warning:
        logging.warning(
            "Given sequence lengths are longer than necessary for some "
            "evaluation inputs or targets, resulting in wasted computation. "
            "Consider passing `None` for `sequence_length` to have them be "
            "automatically computed.\n Got: %s,\n Max Lengths: %s",
            sequence_length, max_lengths)
      elif log_same_warning:
        logging.warning(
            "Given sequence lengths *may be* insufficient for some evaluation "
            "inputs or targets. Such sequences will be truncated to fit, "
            "likely leading to sub-optimal results. Consider passing `None` "
            "for `sequence_length` to have them be automatically computed.\n "
            " Got: %s,\n Max Lengths: %s", sequence_length, max_lengths)

    self._cached_model_datasets = {}

    if feature_converter.pack:
      raise ValueError("During evaluation, packing can't be used.")
    # Convert the task features to model features
    for task in self._all_tasks:
      eval_ds = feature_converter(cached_task_datasets[task.name],
                                  sequence_length)

      # The eval dataset is enumerated to ensure that the order is preserved
      # throughout the entire evaluation process.
      self._cached_model_datasets[task.name] = eval_ds.enumerate()

    self._cached_targets = cached_targets
    self._cached_task_datasets = cached_task_datasets
    self._model_feature_shapes = {
        k: tuple(spec.shape)
        for k, spec in eval_ds.element_spec.items()
        if spec.shape.rank > 0
    }

    if logger_cls and not log_dir:
      raise ValueError(
          "'log_dir' must be provided to `Evaluator` if `logger_cls` is "
          "non-empty.")
    self._loggers = tuple(cls(output_dir=log_dir) for cls in logger_cls)  # pytype:disable=not-instantiable

  def evaluate(
      self,
      *,
      do_inference: bool = False,
      step: Optional[int] = None,
      predict_fn: FewShotPredictFnCallable,
      score_fn: Optional[seqio.evaluation.ScoreFnCallable] = None,
      predict_with_aux_fn: Optional[seqio.evaluation.PredictFnCallable] = None,
  ):
    """Predict and score self.eval_tasks with prediction service.

    Args:
      do_inference: whether always to do inference.
      step: an optional step number of the current evaluation. If unspecified, a
        dummy value of -1 will be used.
      predict_fn: a user-defined function, which takes in a tf.data.Dataset and
        outputs the sequence of predicted tokens. Only called if predict metrics
        exist for the tasks.
      score_fn: a user-defined function, which takes in a tf.data.Dataset and
        outputs the log likelihood score of the targets. Only called if score
        metrics exist for the task.
      predict_with_aux_fn: a user-defined function that has exactly the same
        behaviour as predict_fn, except that it also returns a dictionary of
        auxiliary values. Only called if predict_with_aux metrics exist for the
        tasks.

    Returns:
      metrics: a Future containing a mapping from task name to computed metrics,
        or None if `compute_metrics` is False.
      predicted_tokens: a mapping from task name to the output tokens
        from `predict_fn`, for tasks that have `predict_metric_fns`.
      scores: a mapping from task name to the output scores from
        `score_fn` for tasks that have `score_predict_fns`.
    """
    if score_fn is not None:
      raise NotImplementedError

    if predict_with_aux_fn is not None:
      raise NotImplementedError

    # Maps task.name to a list of sequences of output tokens from the model.
    all_output_tokens = {}
    all_output_scores = {}
    all_aux_values = {}

    for task in self.all_tasks:
      logging.info("Evaluating %s", task.name)

      if task.predict_metric_fns or do_inference:
        tokens, _ = _extract_tokens(
            self._cached_model_datasets[task.name], predict_fn, task)
        all_output_tokens[task.name] = tokens
        all_aux_values[task.name] = {}

    def compute_metrics_fn():
      tick = time.time()
      metrics = self._compute_metrics(all_output_tokens, all_output_scores,
                                      all_aux_values, step)
      logging.info("Time computing metrics: %f secs.", time.time() - tick)
      return metrics

    all_metrics = compute_metrics_fn()

    return all_metrics, all_output_tokens, all_output_scores

  def __del__(self):
    pass

  @property
  def all_tasks(self) -> Sequence[Task]:
    return self._all_tasks


def _extract_tokens(cached_model_dataset, predict_fn, task):
  """Extracts tokens and aux scores from a cached dataset."""
  predict_fn_result = predict_fn(cached_model_dataset, task=task)

  all_aux_values = {}
  indices_and_tokens = predict_fn_result
  _, tokens = zip(*sorted(indices_and_tokens, key=lambda x: x[0]))

  return tokens, all_aux_values


def get_targets_and_examples(
    tasks: Sequence[Task],
    dataset_fn: Callable[[Task], tf.data.Dataset],
    sequence_dims: Mapping[str, int],
    num_examples: Optional[int] = None,
    use_memory_cache: bool = True,
    target_field_name: str = "targets"
) -> Tuple[Mapping[str, Any], Mapping[str, tf.data.Dataset], Mapping[str, int]]:
  """Get targets, cached datasets, and maximum sequence lengths per feature.

  Args:
    tasks: tasks objects to get targets and examples for.
    dataset_fn: function, returns the dataset from the task object.
    sequence_dims: dict of feature names to their sequence dimension.
    num_examples: an optional maximum number of examples to take from the
      beginning of each task dataset.
    use_memory_cache: whether to use tf.data.Dataset#cache. may cause memory
      issues for large datasets.
    target_field_name: Field name of the target in the input dataset examples.

  Returns:
    cached_targets: unpreprocessed targets for each task
    cached_task_datasets: cached datasets for each task, with cardinality set
    max_sequence_length: maximum sequence lengths for inputs and targets across
      all tasks.
  """
  # Pre-load in all of the targets once before entering continuous eval loop
  cached_targets = {}
  cached_task_datasets = {}
  max_sequence_length = {k: 0 for k in tasks[0].output_features.keys()}

  for task in tasks:
    assert max_sequence_length.keys() == task.output_features.keys(), (
        "all tasks must have the same features")

  for task in tasks:
    ds = dataset_fn(task)
    if num_examples:
      ds = ds.take(num_examples)
    if use_memory_cache:
      ds = ds.cache()

    targets = []

    for ex in tqdm.tqdm(tfds.as_numpy(ds)):
      for k in max_sequence_length:
        sequence_dim = sequence_dims.get(k, 0)
        sequence_length = ex[k].shape[sequence_dim]
        max_sequence_length[k] = max(max_sequence_length[k], sequence_length)

      # Create list of postprocessed targets
      pretokenized_target_field_name = target_field_name + "_pretokenized"
      if pretokenized_target_field_name in ex:
        target = ex[pretokenized_target_field_name]
      else:
        target = task.output_features[target_field_name].vocabulary.decode(
            [int(x) for x in ex[target_field_name]])
      if isinstance(target, bytes):
        utf8_target = None
        while utf8_target is None:
          try:
            utf8_target = target.decode("utf-8")
          except UnicodeDecodeError:
            target = target[:-1]
        target = utf8_target
      targets.append(task.postprocess_fn(target, example=ex, is_target=True))

    cached_targets[task.name] = targets
    cached_task_datasets[task.name] = ds.apply(
        tf.data.experimental.assert_cardinality(len(targets)))

  return cached_targets, cached_task_datasets, max_sequence_length
