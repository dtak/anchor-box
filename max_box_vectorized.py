import numpy as np
import tensorflow as tf
import timeit


@tf.function(experimental_relax_shapes=True,
             input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None], dtype=tf.float32)])
def points_within(points, l, u):
  print('Tracing points_within')
  return tf.boolean_mask(points,
                         tf.math.logical_and(
                           tf.reduce_all(points >= l, axis=1),
                           tf.reduce_all(points <= u, axis=1)), axis=0)


@tf.function(experimental_relax_shapes=True,
             input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None], dtype=tf.float32)])
def points_within_strict(points, l, u):
  print('Tracing points_within_strict')
  return tf.boolean_mask(points,
                         tf.math.logical_and(
                           tf.reduce_all(points > l, axis=1),
                           tf.reduce_all(points < u, axis=1)), axis=0)


@tf.function(experimental_relax_shapes=True,
             input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
def count_within(points, ls, us):
  print('Tracing count_within')
  return tf.reduce_sum(tf.cast(tf.math.logical_and(
    tf.reduce_all(points[None, :, :] >= ls[:, None, :], axis=2),
    tf.reduce_all(points[None, :, :] <= us[:, None, :], axis=2)), dtype=tf.int32), axis=1)


@tf.function(experimental_relax_shapes=True,
             input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
def count_within_strict(points, ls, us):
  print('Tracing count_within_strict')
  return tf.reduce_sum(tf.cast(tf.math.logical_and(
    tf.reduce_all(points[None, :, :] > ls[:, None, :], axis=2),
    tf.reduce_all(points[None, :, :] < us[:, None, :], axis=2)), dtype=tf.int32), axis=1)


@tf.function(experimental_relax_shapes=True)
def find_split(l_max, u_min, positive_points_within, negative_points_within):
  print('Tracing find_split')
  dim = tf.shape(l_max)[0]
  band_matrix = 1. - tf.linalg.band_part(tf.ones(shape=[dim, dim]), 0, -1)  # 1. if j < i, 0. otherwise
  sub_l_maxes = tf.where(band_matrix > 0., tf.math.minimum(l_max, negative_points_within[:, None, :]),
                         l_max)
  sub_u_mins = tf.where(band_matrix > 0., tf.math.maximum(u_min, negative_points_within[:, None, :]), u_min)

  # If we make the cut below the Kth negative point along axis L, these positive points are capturable.
  potential_points_below = negative_points_within[:, None, :] >= positive_points_within[None, :, :]  # K x _ x L
  potential_counts_below = tf.reduce_sum(tf.cast(potential_points_below, tf.int32), axis=1)
  sub_below_l_mins = tf.reduce_min(tf.where(potential_points_below[:, :, :, None],
                                            positive_points_within[None, :, None, :],
                                            np.infty),
                                   axis=1)
  sub_below_u_maxes = tf.reduce_max(tf.where(potential_points_below[:, :, :, None],
                                             positive_points_within[None, :, None, :],
                                             -np.infty),
                                    axis=1)
  potential_points_above = negative_points_within[:, None, :] <= positive_points_within[None, :, :]
  potential_counts_above = tf.reduce_sum(tf.cast(potential_points_above, tf.int32), axis=1)
  sub_above_l_mins = tf.reduce_min(tf.where(potential_points_above[:, :, :, None],
                                            positive_points_within[None, :, None, :],
                                            np.infty),
                                   axis=1)
  sub_above_u_maxes = tf.reduce_max(tf.where(potential_points_above[:, :, :, None],
                                             positive_points_within[None, :, None, :],
                                             -np.infty),
                                    axis=1)
  subproblems_l_mins = tf.concat([sub_below_l_mins, sub_above_l_mins], axis=1)
  subproblems_u_maxes = tf.concat([sub_below_u_maxes, sub_above_u_maxes], axis=1)
  subproblems_l_maxes = tf.concat([sub_l_maxes, sub_l_maxes], axis=1)
  subproblems_u_mins = tf.concat([sub_u_mins, sub_u_mins], axis=1)
  subproblem_potentials = tf.concat([potential_counts_below, potential_counts_above], axis=1)

  subproblem_feasibility = tf.reduce_all(
    tf.stack([
      tf.reduce_all(subproblems_l_mins <= subproblems_u_maxes, axis=2),
      tf.reduce_all(subproblems_l_mins <= subproblems_l_maxes, axis=2),
      tf.reduce_all(subproblems_u_mins <= subproblems_u_maxes, axis=2),
      ~tf.reduce_any(tf.math.logical_and(
        tf.reduce_all(negative_points_within[:, None, None, :] > subproblems_l_maxes[None, :, :, :], axis=-1),
        tf.reduce_all(negative_points_within[:, None, None, :] < subproblems_u_mins[None, :, :, :], axis=-1)),
        axis=0)
    ], axis=0),
    axis=0
  )

  split_potentials = tf.reduce_max(tf.where(subproblem_feasibility, subproblem_potentials, -1), axis=1)
  best_split_index = tf.argmin(split_potentials)
  potential = split_potentials[best_split_index]
  split_point = negative_points_within[best_split_index]
  return split_point, potential


@tf.function(experimental_relax_shapes=True)
def do_split(problem_index,
             problems_l_min,
             problems_l_max,
             problems_u_min,
             problems_u_max,
             positive_points,
             negative_points,
             problems_bound,
             problems_negative_within,
             problems_num_splits,
             n_dims):
  print('Tracing do_split')
  l_max = tf.gather(problems_l_max, problem_index)
  l_min = tf.gather(problems_l_min, problem_index)
  u_max = tf.gather(problems_u_max, problem_index)
  u_min = tf.gather(problems_u_min, problem_index)

  negative_points_within = points_within_strict(negative_points, l_min, u_max)
  positive_points_within = points_within(positive_points, l_min, u_max)

  split_point, _ = find_split(l_max, u_min, positive_points_within, negative_points_within)

  band_matrix = 1. - tf.linalg.band_part(tf.ones(shape=[n_dims, n_dims]), 0,
                                         -1)  # 1. if j < i, 0. otherwise
  sub_l_maxes = tf.where(band_matrix > 0., tf.math.minimum(l_max, split_point[None, :]),
                         l_max)
  sub_u_mins = tf.where(band_matrix > 0., tf.math.maximum(u_min, split_point[None, :]), u_min)

  potential_points_below = split_point[None, :] >= positive_points_within[:, :]
  sub_below_l_mins = tf.reduce_min(tf.where(potential_points_below[:, :, None],
                                            positive_points_within[:, None, :],
                                            np.infty),
                                   axis=0)
  sub_below_u_maxes = tf.reduce_max(tf.where(potential_points_below[:, :, None],
                                             positive_points_within[:, None, :],
                                             -np.infty),
                                    axis=0)
  potential_points_above = split_point[None, :] <= positive_points_within[:, :]
  sub_above_l_mins = tf.reduce_min(tf.where(potential_points_above[:, :, None],
                                            positive_points_within[:, None, :],
                                            np.infty),
                                   axis=0)
  sub_above_u_maxes = tf.reduce_max(tf.where(potential_points_above[:, :, None],
                                             positive_points_within[:, None, :],
                                             -np.infty),
                                    axis=0)
  subproblems_l_mins = tf.concat([sub_below_l_mins, sub_above_l_mins], axis=0)
  subproblems_u_maxes = tf.concat([sub_below_u_maxes, sub_above_u_maxes], axis=0)
  subproblems_l_maxes = tf.concat([sub_l_maxes, sub_l_maxes], axis=0)
  subproblems_u_mins = tf.concat([sub_u_mins, sub_u_mins], axis=0)

  subproblem_feasibility = tf.reduce_all(
    tf.stack([
      tf.reduce_all(subproblems_l_mins <= subproblems_u_maxes, axis=1),
      tf.reduce_all(subproblems_l_mins <= subproblems_l_maxes, axis=1),
      tf.reduce_all(subproblems_u_mins <= subproblems_u_maxes, axis=1),
      ~tf.reduce_any(tf.math.logical_and(
        tf.reduce_all(negative_points_within[:, None, :] > subproblems_l_maxes[None, :, :], axis=-1),
        tf.reduce_all(negative_points_within[:, None, :] < subproblems_u_mins[None, :, :], axis=-1)),
        axis=0)
    ], axis=0),
    axis=0
  )

  subproblems_l_mins = tf.boolean_mask(subproblems_l_mins, subproblem_feasibility, axis=0)
  subproblems_l_maxes = tf.boolean_mask(subproblems_l_maxes, subproblem_feasibility, axis=0)
  subproblems_u_mins = tf.boolean_mask(subproblems_u_mins, subproblem_feasibility, axis=0)
  subproblems_u_maxes = tf.boolean_mask(subproblems_u_maxes, subproblem_feasibility, axis=0)

  subproblem_bounds = tf.zeros(shape=[tf.shape(subproblems_l_mins)[0]], dtype=tf.int32)
  for i in tf.range(tf.shape(subproblems_l_mins)[0]):
    l_max = tf.gather(subproblems_l_maxes, i)
    l_min = tf.gather(subproblems_l_mins, i)
    u_max = tf.gather(subproblems_u_maxes, i)
    u_min = tf.gather(subproblems_u_mins, i)
    sub_positive_points_within = points_within(positive_points_within, l_min, u_max)
    sub_negative_points_within = points_within_strict(negative_points_within, l_min, u_max)
    if tf.math.equal(tf.shape(sub_negative_points_within)[0], 0):
      bound = tf.shape(sub_positive_points_within)[0]
    else:
      _, bound = find_split(l_max, u_min, sub_positive_points_within, sub_negative_points_within)
    subproblem_bounds = tf.tensor_scatter_nd_update(subproblem_bounds, [[i]], [bound])

  # Discard impossible subproblems after find_split
  subproblems_l_mins = tf.boolean_mask(subproblems_l_mins, subproblem_bounds > 0, axis=0)
  subproblems_l_maxes = tf.boolean_mask(subproblems_l_maxes, subproblem_bounds > 0, axis=0)
  subproblems_u_mins = tf.boolean_mask(subproblems_u_mins, subproblem_bounds > 0, axis=0)
  subproblems_u_maxes = tf.boolean_mask(subproblems_u_maxes, subproblem_bounds > 0, axis=0)
  subproblem_bounds = tf.boolean_mask(subproblem_bounds, subproblem_bounds > 0, axis=0)

  subproblem_num_splits = (tf.zeros(shape=[tf.shape(subproblems_l_mins)[0]], dtype=tf.int32)
                           + tf.gather(problems_num_splits, problem_index)
                           + 1)
  subproblem_negative_within = count_within_strict(negative_points_within, subproblems_l_mins, subproblems_u_maxes)

  problems_l_min.assign(tf.concat([problems_l_min[:problem_index, :],
                                   problems_l_min[problem_index + 1:, :],
                                   subproblems_l_mins], axis=0))
  problems_l_max.assign(tf.concat([problems_l_max[:problem_index, :],
                                   problems_l_max[problem_index + 1:, :],
                                   subproblems_l_maxes], axis=0))
  problems_u_min.assign(tf.concat([problems_u_min[:problem_index, :],
                                   problems_u_min[problem_index + 1:, :],
                                   subproblems_u_mins], axis=0))
  problems_u_max.assign(tf.concat([problems_u_max[:problem_index, :],
                                   problems_u_max[problem_index + 1:, :],
                                   subproblems_u_maxes], axis=0))
  problems_bound.assign(tf.concat([problems_bound[:problem_index],
                                   problems_bound[problem_index + 1:],
                                   subproblem_bounds], axis=0))
  problems_negative_within.assign(tf.concat([problems_negative_within[:problem_index],
                                             problems_negative_within[problem_index + 1:],
                                             subproblem_negative_within], axis=0))
  problems_num_splits.assign(tf.concat([problems_num_splits[:problem_index],
                                        problems_num_splits[problem_index + 1:],
                                        subproblem_num_splits], axis=0))


@tf.function(experimental_relax_shapes=True)
def current_solution(problems_l_min, problems_u_max, problems_bound, problems_negative_within):
  problem_index = tf.argmax(tf.where(problems_negative_within == 0, problems_bound, -1))

  l_min = tf.gather(problems_l_min, problem_index)
  u_max = tf.gather(problems_u_max, problem_index)

  bound = tf.gather(problems_bound, problem_index)

  return l_min, u_max, bound


@tf.function(experimental_relax_shapes=True)
def select_problem(deep_dive, problems_bound, problems_negative_within, problems_num_splits):
  if deep_dive:
    k = tf.argmax(
      tf.where(problems_num_splits == tf.reduce_max(problems_num_splits), problems_bound, -1))
    return tf.argmax(
      tf.where(problems_num_splits == tf.reduce_max(problems_num_splits), problems_bound, -1))
  else:
    best_so_far = tf.reduce_max(tf.where(problems_negative_within == 0, problems_bound, -1))
    return tf.argmax(tf.where(tf.math.logical_and(
      problems_negative_within > 0, problems_bound > best_so_far),
      tf.cast(problems_bound, tf.float32)
      / tf.cast(problems_negative_within, dtype=tf.float32),
      -1))


@tf.function(experimental_relax_shapes=True)
def deep_dive(problems_l_min,
              problems_l_max,
              problems_u_min,
              problems_u_max,
              positive_points,
              negative_points,
              problems_bound,
              problems_negative_within,
              problems_num_splits,
              n_dims):
  while tf.reduce_all(problems_negative_within > 0):
    problem_index = select_problem(True, problems_bound, problems_negative_within, problems_num_splits)
    do_split(problem_index,
             problems_l_min,
             problems_l_max,
             problems_u_min,
             problems_u_max,
             positive_points,
             negative_points,
             problems_bound,
             problems_negative_within,
             problems_num_splits,
             n_dims)


@tf.function(experimental_relax_shapes=True)
def solve(maxiter,
          problems_l_min,
          problems_l_max,
          problems_u_min,
          problems_u_max,
          positive_points,
          negative_points,
          problems_bound,
          problems_negative_within,
          problems_num_splits,
          n_dims):
  deep_dive(problems_l_min,
            problems_l_max,
            problems_u_min,
            problems_u_max,
            positive_points,
            negative_points,
            problems_bound,
            problems_negative_within,
            problems_num_splits,
            n_dims)
  for _ in tf.range(maxiter):
    problem_index = select_problem(False, problems_bound, problems_negative_within, problems_num_splits)
    best_so_far = tf.reduce_max(tf.where(problems_negative_within == 0, problems_bound, -1))
    if tf.math.logical_and(tf.gather(problems_negative_within, problem_index) > 0,
                           tf.gather(problems_bound, problem_index) > best_so_far):
      do_split(problem_index,
               problems_l_min,
               problems_l_max,
               problems_u_min,
               problems_u_max,
               positive_points,
               negative_points,
               problems_bound,
               problems_negative_within,
               problems_num_splits,
               n_dims)
    else:
      break

  return current_solution(problems_l_min, problems_u_max, problems_bound, problems_negative_within)


@tf.function(experimental_relax_shapes=True)
def add_negative(points, problems_l_min, problems_u_max, negative_points, problems_negative_within):
  problems_negative_within.assign_add(count_within_strict(points,
                                                          problems_l_min,
                                                          problems_u_max))
  negative_points.assign(tf.concat([negative_points, points], axis=0))


@tf.function(experimental_relax_shapes=True)
def post_expand(l, u, negative_points, universe_min, universe_max, n_dims):
  print('Tracing post_expand')
  l_already_done = tf.zeros(shape=(n_dims,), dtype=tf.bool)
  u_already_done = tf.zeros(shape=(n_dims,), dtype=tf.bool)

  for _ in tf.range(2 * n_dims):
    within_l = (negative_points > l)
    within_u = (negative_points < u)
    l_bounding = tf.math.logical_and(tf.reduce_sum(tf.cast(within_l, tf.int32), axis=1) == n_dims - 1,
                                     tf.reduce_all(within_u, axis=1))
    u_bounding = tf.math.logical_and(tf.reduce_sum(tf.cast(within_u, tf.int32), axis=1) == n_dims - 1,
                                     tf.reduce_all(within_l, axis=1))
    l_dim = tf.argmin(within_l, axis=1)
    u_dim = tf.argmin(within_u, axis=1)
    l_extension = tf.where(l_bounding,
                           (tf.gather_nd(u, l_dim[:, None]) - tf.gather_nd(negative_points, l_dim[:, None],
                                                                           batch_dims=1)) / (
                               tf.gather_nd(u, l_dim[:, None]) - tf.gather_nd(l, l_dim[:, None])),
                           np.infty)  # N
    u_extension = tf.where(u_bounding,
                           (tf.gather_nd(negative_points, u_dim[:, None], batch_dims=1) - tf.gather_nd(l,
                                                                                                       u_dim[:,
                                                                                                       None])) / (
                               tf.gather_nd(u, u_dim[:, None]) - tf.gather_nd(l, u_dim[:, None])),
                           np.infty)  # N
    l_universe_bound = (u - universe_min) / (u - l)
    u_universe_bound = (universe_max - l) / (u - l)
    l_max_extension = tf.where(l_already_done, 0.,
                               tf.tensor_scatter_nd_min(l_universe_bound, l_dim[:, None], l_extension))
    u_max_extension = tf.where(u_already_done, 0.,
                               tf.tensor_scatter_nd_min(u_universe_bound, u_dim[:, None], u_extension))

    if tf.math.greater(tf.reduce_max(l_max_extension), tf.reduce_max(u_max_extension)):
      chosen_dim = tf.argmax(l_max_extension)
      # chosen_dim = tf.random.uniform(shape=[], minval=0, maxval=tf.cast(tf.shape(l_max_extension)[0], dtype=tf.int64), dtype=tf.int64)
      if tf.reduce_any(tf.logical_and(l_dim == chosen_dim, l_bounding)):
        bounding_point = tf.argmin(tf.where(tf.logical_and(l_dim == chosen_dim, l_bounding), l_extension, np.infty))
        l = tf.tensor_scatter_nd_update(l, [[chosen_dim]],
                                        tf.gather_nd(negative_points, [[bounding_point, chosen_dim]]))
      else:
        l = tf.tensor_scatter_nd_update(l, [[chosen_dim]], tf.gather(universe_min, [chosen_dim]))
      l_already_done = tf.tensor_scatter_nd_update(l_already_done, [[chosen_dim]], [True])
    else:
      chosen_dim = tf.argmax(u_max_extension)
      # chosen_dim = tf.random.uniform(shape=[], minval=0, maxval=tf.cast(tf.shape(u_max_extension)[0], dtype=tf.int64), dtype=tf.int64)
      if tf.reduce_any(tf.logical_and(u_dim == chosen_dim, u_bounding)):
        bounding_point = tf.argmin(tf.where(tf.logical_and(u_dim == chosen_dim, u_bounding), u_extension, np.infty))
        u = tf.tensor_scatter_nd_update(u, [[chosen_dim]],
                                        tf.gather_nd(negative_points, [[bounding_point, chosen_dim]]))
      else:
        u = tf.tensor_scatter_nd_update(u, [[chosen_dim]], tf.gather(universe_max, [chosen_dim]))
      u_already_done = tf.tensor_scatter_nd_update(u_already_done, [[chosen_dim]], [True])
  return l, u


class MaxBoxProblem(object):
  def __init__(self, test_point, positive_points, negative_points, universe_min, universe_max):
    self.n_dims = test_point.shape[0]
    self.test_point = test_point
    self.positive_points = tf.concat([test_point[None, :], positive_points], axis=0)
    self.negative_points = tf.Variable(negative_points, shape=[None, self.n_dims], trainable=False)
    self.universe_min = universe_min
    self.universe_max = universe_max
    self.problems_l_min = tf.Variable(tf.reduce_min(positive_points, axis=0)[None, :], shape=[None, self.n_dims],
                                      trainable=False)
    self.problems_l_max = tf.Variable(test_point[None, :], shape=[None, self.n_dims], trainable=False)
    self.problems_u_min = tf.Variable(test_point[None, :], shape=[None, self.n_dims], trainable=False)
    self.problems_u_max = tf.Variable(tf.reduce_max(positive_points, axis=0)[None, :], shape=[None, self.n_dims],
                                      trainable=False)
    self.problems_bound = tf.Variable(tf.zeros(shape=[1], dtype=tf.int32)
                                      + tf.shape(positive_points)[0], shape=[None], trainable=False)
    self.problems_negative_within = tf.Variable(count_within(self.negative_points,
                                                             self.problems_l_min,
                                                             self.problems_u_max),
                                                shape=[None], trainable=False)
    self.problems_num_splits = tf.Variable(tf.zeros(shape=[1], dtype=tf.int32), shape=[None], trainable=False)

  def solve(self, maxiter):
    return solve(maxiter,
                 self.problems_l_min,
                 self.problems_l_max,
                 self.problems_u_min,
                 self.problems_u_max,
                 self.positive_points,
                 self.negative_points,
                 self.problems_bound,
                 self.problems_negative_within,
                 self.problems_num_splits,
                 self.n_dims)

  def add_negative(self, points):
    add_negative(points, self.problems_l_min, self.problems_u_max, self.negative_points, self.problems_negative_within)

  def post_expand(self, l, u):
    return post_expand(l, u, self.negative_points, self.universe_min, self.universe_max, self.n_dims)
