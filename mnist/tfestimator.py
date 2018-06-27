import numpy as np
import tensorflow as tf


tf.app.flags.DEFINE_integer('batch_size', 32, 'training batch size')
tf.app.flags.DEFINE_integer('steps', 100, 'training steps')
tf.app.flags.DEFINE_string(
    'checkpoint_dir', './models/ckpt', 'path to save checkpoints')
tf.app.flags.DEFINE_string('save_dir', './models/pb',
                           'path to save a model for serving')
FLAGS = tf.app.flags.FLAGS


INPUT_FEATURE = 'image'
N_CLASSES = 10


def model_fn(features, labels, mode, params):
    x = features[INPUT_FEATURE]
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
    logits = tf.layers.dense(x, N_CLASSES)

    prediction_classes = tf.argmax(logits, axis=-1)
    predictions = dict(
        class_ids=prediction_classes,
        probabilities=tf.nn.softmax(
            logits, axis=-1, name='probabilities_tensor'),
        logits=logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            export_outputs={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions),
            })

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


def serving_input_receiver_fn():
    receiver_tensors = {
        INPUT_FEATURE: tf.placeholder(tf.string, [None]),
    }
    image = tf.map_fn(lambda x: tf.image.decode_image(x, channels=1), receiver_tensors[INPUT_FEATURE], dtype=tf.uint8)
    image.set_shape([None, None, None, 1])
    image = tf.cast(image, dtype=tf.int32)
    features = {
        INPUT_FEATURE: tf.image.resize_images(image, [28, 28]),
    }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors, features=features)


def main(_):
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
    eval_data = eval_data.reshape(eval_data.shape[0], 28, 28, 1)

    training_config = tf.estimator.RunConfig(model_dir=FLAGS.checkpoint_dir)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=FLAGS.checkpoint_dir, config=training_config)

    logging_tensors = {'probabilities': 'probabilities_tensor'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=logging_tensors, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE: train_data},
        y=train_labels,
        batch_size=FLAGS.batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE: eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    train_spec = tf.estimator.TrainSpec(
        train_input_fn, max_steps=FLAGS.steps, hooks=[logging_hook])
    eval_spec = tf.estimator.EvalSpec(eval_input_fn)

    tf.estimator.train_and_evaluate(
        estimator, train_spec=train_spec, eval_spec=eval_spec)

    estimator.export_savedmodel(
        FLAGS.save_dir, serving_input_receiver_fn=serving_input_receiver_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
