import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist


def model_fn(features, labels, mode, params):
    x = tf.feature_column.input_layer(features, params['feature_columns'])
    x = tf.reshape(x, (-1, 28, 28, 1))
    x = tf.layers.conv2d(x, 32, 3, padding='same', activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 3, 2, padding='same')
    x = tf.layers.batch_normalization(
        x, training=mode == tf.estimator.ModeKeys.TRAIN)
    x = tf.layers.conv2d(x, 64, 3, padding='same', activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 3, 2, padding='same')
    x = tf.layers.batch_normalization(
        x, training=mode == tf.estimator.ModeKeys.TRAIN)
    x = tf.layers.conv2d(x, 128, 3, padding='same', activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 3, 2, padding='same')
    x = tf.layers.batch_normalization(
        x, training=mode == tf.estimator.ModeKeys.TRAIN)
    x = tf.reshape(x, (-1, 4 * 4 * 128))
    logits = tf.layers.dense(x, params['n_classes'])

    prediction_classes = tf.argmax(logits, axis=-1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = dict(
            class_ids=prediction_classes,
            probabilities=tf.nn.softmax(logits, axis=-1),
            logits=logits)
        return predictions

    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    accuracy = tf.metrics.accuracy(labels, prediction_classes, name='acc_op')
    metrics = dict(accuracy=accuracy)
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def input_fn(features, labels, batch_size, shuffle=True, repeat=False):
    labels = labels.astype('int32')
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(10000)
    if repeat:
        dataset = dataset.repeat()
    return dataset.batch(batch_size)


def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=500000)
    parser.add_argument('--mnist-dir', default='./MNIST_data')
    parser.add_argument('--model-dir', default='./model')
    args = parser.parse_args(argv[1:])

    mnist_data = mnist.input_data.read_data_sets(args.mnist_dir)
    feature_columns = [tf.feature_column.numeric_column(
        key='image', shape=(784,))]

    config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        save_checkpoints_secs=10,
    )
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config, params=dict(
        feature_columns=feature_columns, hidden_units=[128, 64, 32], n_classes=10))

    def train_input_fn():
        return input_fn(dict(image=mnist_data.train.images), mnist_data.train.labels, args.batch_size, repeat=True)

    def test_input_fn():
        return input_fn(dict(image=mnist_data.test.images), mnist_data.test.labels, args.batch_size)

    experiment = tf.contrib.learn.Experiment(estimator, train_input_fn, test_input_fn,
                                             train_steps=args.steps,
                                             min_eval_frequency=1)
    experiment.train_and_evaluate()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
