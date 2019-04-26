""" A neural chatbot using sequence to sequence model with attentional decoder.

This is based on Google Translate (PTB_LSTM) Tensorflow model

Sequence to sequence model by Cho et al.(2014)
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu

This file contains the hyperparameters for the model.
"""

# Parameters for processing the dataset
DATA_PATH = 'data/cornell movie-dialogs corpus'
CONVO_FILE = 'movie_conversations.txt'  # contains the structure of the conversations
                                        # list of the utterances that make the conversation, in chronological order:
                                        # ['lineID1','lineID2',É,'lineIDN']
                                        # has to be matched with movie_lines.txt to reconstruct the actual content

LINE_FILE = 'movie_lines.txt'       # contains the actual text of each utterance
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = 'processed'
CPT_PATH = 'checkpoints'    # If you want to start training from scratch, please delete all the checkpoints in the checkpoints folder.

THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 25000

BUCKETS = [(19, 19), (28, 28), (33, 33), (40, 43), (50, 53), (60, 63)]
""" Bucketing:

● Avoid too much padding that leads to extraneous computation
● Group sequences of similar lengths into the same buckets
● Create a separate subgraph for each bucket
● In theory, can use for v1.0:
    tf.contrib.training.bucket_by_sequence_length(max_length, examples, batch_size, 
    bucket_boundaries, capacity=2 * batch_size, dynamic_pad=True)
● In practice, use the bucketing algorithm used in TensorFlow’s translate model (because we’re using v0.12)

Basically, data is arranged into different inputs 
- If the length of the sequence (utterance) is shorter than [0] for encode or [1] for decode, then put it in this group
"""
CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "),
                ("don ' t ", "do n't "), ("didn ' t ", "did n't "), ("doesn ' t ", "does n't "),
                ("can ' t ", "ca n't "), ("shouldn ' t ", "should n't "), ("wouldn ' t ", "would n't "),
                ("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

NUM_LAYERS = 3
HIDDEN_SIZE = 256   # -no of neurons
BATCH_SIZE = 64

LR = 0.5
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 512
ENC_VOCAB = 24404
DEC_VOCAB = 24593
