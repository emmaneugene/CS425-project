from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(11)
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #suppress info, warning
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #suppress info, warning
import tensorflow_hub as hub
from datetime import datetime

import emotional_classifier.bert_files.run_classifier as run_classifier
import emotional_classifier.bert_files.optimization as optimization
import emotional_classifier.bert_files.tokenization as tokenization



''' CONSTANTS '''
OUTPUT_DIR = 'bert_files/output'
MAX_SEQ_LENGTH = 128
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1
len_train_features = 39585
num_train_steps = int(len_train_features / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

label_list = [0, 1, 2, 3, 4]




''' FUNCTIONS '''

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
  """Creates a classification model."""
  #i guess to adapt this such that i can find 32 labels, just need to pass in the correct labels and num_labels into this function 

  input_ids = tf.cast(input_ids, tf.int32)
  input_mask = tf.cast(input_mask, tf.int32)
  segment_ids = tf.cast(segment_ids, tf.int32)

  bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = bert_outputs["pooled_output"]

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)


def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      train_op = optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics. 
      def metric_fn(label_ids, predicted_labels):
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        recall = tf.metrics.recall(
            label_ids,
            predicted_labels)
        precision = tf.metrics.precision(
            label_ids,
            predicted_labels) 
        return {
            "eval_accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }

      eval_metrics = metric_fn(label_ids, predicted_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn


def map_emotion_to_response(emotion):
    ''' Returns the rules based response given the emotion '''
    emotion_map = {
        "Joy": "I'm so happy for you! ",
        "Sadness": "I am sorry to hear that. ",
        "Anger": "Calm down, everything will be alright. ",
        "Fear": "That's quite scary! ",
        "Neutral": ""
    }

    return emotion_map.get(emotion, "I'm not sure how you are feeling.")



def load_model():
  ''' 
    Imports the fine-tuned model that we have already previously trained. 
    Returns the loaded model and the tokenizer for the input.
  '''
  tokenizer = create_tokenizer_from_hub_module()

  model_fn = model_fn_builder(
    num_labels=len(label_list),
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps)

  loaded = tf.estimator.Estimator(
      model_fn=model_fn, 
      model_dir=OUTPUT_DIR,
      params={"batch_size": BATCH_SIZE}
  )

  return loaded, tokenizer


def getPrediction(in_sentences, loaded, tokenizer):
  ''' 
    Takes in a list of sentences
    Uses the loaded model and tokenizer to predict the integer representing the emotion for each of the sentences.
    Returns a list of integers
  '''
  labels = ["Joy", "Sadness", "Anger", "Fear", "Neutral"]
  input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] 
  input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
  predictions = loaded.predict(predict_input_fn)
  return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]


def generate_reply(sentence, loaded, tokenizer):
  ''' Returns the rules based response for each sentence using the loaded model and the tokenizer '''
  emotion = getPrediction([sentence, ''], loaded, tokenizer)[0][2]
  response = map_emotion_to_response(emotion)

  return response
