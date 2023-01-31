import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
import timeit
from matplotlib import pyplot as plt

from anchor_explanation import Anchor

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('n_dims', 10, 'Data dimensionality.')
flags.DEFINE_integer('test_size', 10, 'Number of points to test on.')
flags.DEFINE_integer('anchor_samples', 100, 'Number of positive points for the max-box algorithm.')
flags.DEFINE_integer('anchor_search', 100, 'Number of search iterations for the max-box algorithm.')
flags.DEFINE_float('purity', 0.99, 'Purity.')
flags.DEFINE_float('confidence', 0.99, 'Confidence that the target purity is achieved.')


def get_oracle_data(n_dims):
  def b(points):
    b.counter += points.shape[0]
    start = timeit.default_timer()
    too_far = tf.reduce_sum(tf.abs(points[:, (n_dims // 2):]), axis=1) >= (n_dims // 2) * 0.5
    b.timer += timeit.default_timer() - start
    return ~too_far
  return b


def main(argv):
  np.random.seed(FLAGS.seed)
  b = get_oracle_data(FLAGS.n_dims)

  universe_min = np.ones(shape=[FLAGS.n_dims]) * -0.5 * FLAGS.n_dims
  universe_max = np.ones(shape=[FLAGS.n_dims]) * 0.5 * FLAGS.n_dims
  anchor = Anchor(FLAGS.purity,
                  FLAGS.confidence,
                  universe_min,
                  universe_max,
                  n_samples=FLAGS.anchor_samples,
                  max_iter=FLAGS.anchor_search)

  for i in range(FLAGS.test_size):
    test_point = np.random.uniform(low=-0.5, high=0.5, size=[FLAGS.n_dims])
    b.counter = 0
    b.timer = 0.
    start = timeit.default_timer()
    volume, _, _, l_and_u = anchor.explain(b, test_point)
    stop = timeit.default_timer()
    time = stop - start - b.timer
    print(f'{i} vol: {volume}, l and u: {l_and_u} ({time} sec)')

if __name__ == '__main__':
  app.run(main)
