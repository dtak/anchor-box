import numpy as np
import math
import tensorflow as tf
import tensorflow_probability as tfp
from max_box_vectorized import MaxBoxProblem
import random
import timeit

@tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None], dtype=tf.float32)])
def points_within(points, l, u):
  return tf.boolean_mask(points, tf.logical_and(tf.reduce_all(points >= l, axis=1), tf.reduce_all(points <= u, axis=1)), axis=0)

class Anchor(object):
  def __init__(self, purity, confidence, universe_min, universe_max, n_samples, max_iter):
    self.purity = purity
    self.confidence = confidence
    self.universe_min = universe_min.astype(np.float32)
    self.universe_max = universe_max.astype(np.float32)
    self.n_samples = n_samples
    self.max_iter = max_iter

  def test_purity(self, f, l, u, confidence):
    samples_needed = math.ceil(np.log(1. - confidence) / np.log(self.purity))
    samples = tf.random.uniform([samples_needed, u.shape[0]], minval=l, maxval=u)
    mask = f(samples)
    results = tf.reduce_all(mask), tf.boolean_mask(samples, mask, axis=0), tf.boolean_mask(samples, ~mask, axis=0)
    return results

  def confidence_generator(self):
    i = 2
    series_sum = 1.9428735027
    while True:
      yield 1. - (1. - self.confidence) * (1. / (i * np.log(i) * np.log(i))) / series_sum
      i += 1

  def approximate_solve(self, f, test_point, l, u, n_samples):
    n_dims = test_point.shape[0]
    universe_min = l
    universe_max = u
    core_l = l
    core_u = u
    problem = None
    for confidence in self.confidence_generator():
      meets_purity, test_positives, test_negatives = self.test_purity(f, l, u, confidence)
      if meets_purity:
        break
      negative_points = test_negatives[:n_samples, :]
      if problem is None:
        assert np.all(l == universe_min)
        assert np.all(u == universe_max)
        positive_points = test_positives[:n_samples, :]
        while positive_points.shape[0] < n_samples:
          samples = tf.random.uniform([n_samples, u.shape[0]], minval=l, maxval=u)
          mask = f(samples)
          if positive_points.shape[0] < n_samples:
            positive_points = tf.concat([positive_points, tf.boolean_mask(samples, mask, axis=0)], axis=0)[:n_samples, :]
        positive_points = tf.concat([test_point[None, :], positive_points], axis=0)
        problem = MaxBoxProblem(test_point, positive_points, negative_points, universe_min, universe_max)
      else:
        problem.add_negative(negative_points)

      if points_within(negative_points, core_l, core_u).shape[0] > 0:
        # print(f'dims {universe_max.shape[0]}')
        core_l, core_u, in_points = problem.solve(maxiter=self.max_iter)
        # print(f'area percent {np.prod(u - l) / np.prod(universe_max - universe_min)} in-points {in_points}')
        if (in_points == 1 or in_points * 20 < n_samples):  # If less than 5% of positive points are captured, then try a larger n_samples. This happens extremely rarely.
          return self.approximate_solve(f, test_point, core_l, core_u, 2 * n_samples)
      # print('expanding')
      l, u = problem.post_expand(core_l, core_u)
    return l, u

  def solve_dimensions(self, f, test_point, dims, l, u):
    transformed_test_point = tf.gather(test_point, dims)
    # print(dims)

    def f_restricted(transformed_points):
      points = tf.transpose(
        tf.tensor_scatter_nd_update(tf.transpose(test_point * tf.ones([transformed_points.shape[0], 1])),
                                    [[dim] for dim in dims],
                                    tf.transpose(transformed_points)))
      return f(points)

    l, u = self.approximate_solve(f_restricted,
                                  transformed_test_point,
                                  l, u,
                                  self.n_samples)

    return np.array(l), np.array(u)

  def explain(self, f, test_point):
    test_point = test_point.astype(np.float32)
    assert f(test_point[None, :])[0]
    d = test_point.shape[0]
    problem_dims = []
    problem_ls = []
    problem_us = []
    initial_order = [i for i in range(d)]
    random.shuffle(initial_order)
    for i in initial_order:
      l, u = self.solve_dimensions(f, test_point, [i],
                                   np.array([self.universe_min[i]]),
                                   np.array([self.universe_max[i]]))
      problem_dims.append([i])
      problem_ls.append(l)
      problem_us.append(u)

    while len(problem_dims) > 1:
      new_problem_dims = []
      new_problem_ls = []
      new_problem_us = []
      for i in range(0, len(problem_dims) - 1, 2):
        dims1 = problem_dims[i]
        dims2 = problem_dims[i + 1]
        # print(f'Merging {dims1} {dims2}')
        l1 = problem_ls[i]
        l2 = problem_ls[i + 1]
        u1 = problem_us[i]
        u2 = problem_us[i + 1]
        for j in range(1, min(len(dims1), len(dims2)) + 1):
          if j <= len(dims1):
            l = tf.concat([l1[:j], l2], axis=0)
            u = tf.concat([u1[:j], u2], axis=0)
            dims = dims1[:j] + dims2
            # print(f'{i}, Dims {dims}')
            l, u = self.solve_dimensions(f, test_point, dims, l, u)
            l1[:j] = l[:j]
            l2[:] = l[j:]
            u1[:j] = u[:j]
            u2[:] = u[j:]
            # print(f'coming out {l} {u}')
            if j == len(dims1):
              break
          if j <= len(dims2):
            l = tf.concat([l1, l2[:j]], axis=0)
            u = tf.concat([u1, u2[:j]], axis=0)
            dims = dims1 + dims2[:j]
            # print(f'{i}, Dims {dims}')
            l, u = self.solve_dimensions(f, test_point, dims, l, u)
            l1[:] = l[:-j]
            l2[:j] = l[-j:]
            u1[:] = u[:-j]
            u2[:j] = u[-j:]
            # print(f'coming out {l} {u}')
            if j == len(dims2):
              break
        new_problem_dims.append(dims)
        new_problem_ls.append(l)
        new_problem_us.append(u)

      if len(problem_dims) % 2 == 1:
        new_problem_dims.append(problem_dims[-1])
        new_problem_ls.append(problem_ls[-1])
        new_problem_us.append(problem_us[-1])

      # Keeping the binary tree balanced
      indices = list(range(len(new_problem_dims)))
      random.shuffle(indices)
      indices = sorted(indices, key=lambda i: len(new_problem_dims[i]))
      interleaved = [indices[i // 2] if i % 2 == 0 else indices[-(i // 2) - 1] for i in range(len(indices))]
      problem_dims = [new_problem_dims[i] for i in interleaved]
      problem_ls = [new_problem_ls[i] for i in interleaved]
      problem_us = [new_problem_us[i] for i in interleaved]
      # problem_dims = new_problem_dims
      # problem_ls = new_problem_ls
      # problem_us = new_problem_us

    l = tf.transpose(tf.tensor_scatter_nd_update(test_point,
                                                 [[dim] for dim in problem_dims[0]],
                                                 problem_ls[0]))
    u = tf.transpose(tf.tensor_scatter_nd_update(test_point,
                                                 [[dim] for dim in problem_dims[0]],
                                                 problem_us[0]))
    # print(f'result {l} {u}')
    dist = tfp.distributions.Uniform(low=tf.cast(l, tf.float32),
                                     high=tf.cast(u, tf.float32))
    def isinside(points):
      return tf.logical_and(tf.reduce_all(points > tf.cast(l, tf.float32), axis=-1),
                     tf.reduce_all(points < tf.cast(u, tf.float32), axis=-1))
    return (tf.reduce_sum(tf.math.log(tf.cast(u, tf.float32) - tf.cast(l, tf.float32))),
            dist.sample,
            isinside,
            (tf.cast(l, tf.float32), tf.cast(u, tf.float32)))
