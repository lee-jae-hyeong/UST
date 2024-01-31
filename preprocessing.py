"""
Author: Subhabrata Mukherjee (submukhe@microsoft.com)
Code for Uncertainty-aware Self-training (UST) for few-shot learning.
"""

from collections import defaultdict

import csv
import logging
import numpy as np
import six
import tensorflow as tf
from datasets import load_dataset, load_metric, load_from_disk

logger = logging.getLogger('UST')

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def generate_sequence_data(MAX_SEQUENCE_LENGTH, data_type, tokenizer, unlabeled=False, do_pairwise=False):
    
  X1 = []
  X2 = []
  y = []
  path = "/content/drive/MyDrive/UPET/ecommerce_cate"
  raw_datasets = load_from_disk(path)
  
  train = raw_datasets['train']
  val = raw_datasets['validation']
  test= raw_datasets['test']


  if data_type == "train":
    dataset=raw_datasets['train']
  elif data_type == "validation":
    dataset=raw_datasets['validation']
  else:
    dataset=raw_datasets['test']
    
  label_count = defaultdict(int)
  # with tf.io.gfile.GFile(input_file, "r") as f:
  #   reader = csv.reader(f, delimiter="\t", quotechar=None)
  for number in range(len(dataset)):
    if len(dataset["sentence"][number]) == 0:
      continue
    X1.append(convert_to_unicode(dataset["sentence"][number]))
    if do_pairwise:
      X2.append(convert_to_unicode(dataset["sentence"][number]))
    if not unlabeled:
        if do_pairwise:
          label = int(convert_to_unicode(dataset["label"][number]))
        else:
          label = int(convert_to_unicode(dataset["label"][number]))
        y.append(label)
        label_count[label] += 1
    else:
        y.append(-1)
    
  if do_pairwise:
    X =  tokenizer(X1, X2, padding=True, truncation=True, max_length = MAX_SEQUENCE_LENGTH)
  else:
    X =  tokenizer(X1, padding=True, truncation=True, max_length = MAX_SEQUENCE_LENGTH)

  for key in label_count.keys():
      logger.info ("Count of instances with label {} is {}".format(key, label_count[key]))

  if "token_type_ids" not in X:
      token_type_ids = np.zeros((len(X["input_ids"]), MAX_SEQUENCE_LENGTH))
  else:
      token_type_ids = np.array(X["token_type_ids"])

  return {"input_ids": np.array(X["input_ids"]), "token_type_ids": token_type_ids, "attention_mask": np.array(X["attention_mask"])}, np.array(y)

