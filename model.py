import time

import numpy as np
import tensorflow as tf

import config

"""
Seq2seq
● Attentional decoder
● Reverse encoder inputs
● Bucketing
● Sampled softmax
● Based on the Google’s vanilla translate model, originally used to translate from English to French
"""

class ChatBotModel:
    def __init__(self, forward_only, batch_size):
        """ forward_only: if set we don't construct the backward pass in the model """
        print("Initialize new model")
        self.fw_only = forward_only
        self.batch_size = batch_size

    def _create_placeholders(self):
        # Feeds for inputs. It's a list of placeholders
        print('Create placeholders')
        # create a list of placeholder inputs for the first index from reverse (i.e 60)
        # creates a list of 60 sequences (tokens) (so batch_size = 60 for first one)
        # note: batch_size is not no of batches
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                               for i in range(config.BUCKETS[-1][0])]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                               for i in range(config.BUCKETS[-1][1] + 1)]   # plus 1 for the last target
        self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                               for i in range(config.BUCKETS[-1][1] + 1)]

        # Our targets are decoder inputs shifted by one (to ignore <GO> symbol) # decoder_inputs would be the target from encoder
        self.targets = self.decoder_inputs[1:]

    def _inference(self):   # Prediction
        print("Create inference")
        # If we use sampled softmax, we need an output projection
        # Sampled softmax only makes sense if we sample less than vocabulary size
        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < config.DEC_VOCAB:
            w = tf.get_variable('proj_w', [config.HIDDEN_SIZE, config.DEC_VOCAB])       # trainable = True
            b = tf.get_variable('proj_b', [config.DEC_VOCAB])       # trainable = True
            self.output_projection = (w, b)

        """
        Sampled Softmax
        ● Avoid the growing complexity of computing the normalization constant
        ● Approximate the negative term of the gradient, by importance sampling with a small number of samples.
        ● At each step, update only the vectors associated with the correct word w and with the sampled words in V’
        ● Once training is over, use the full target vocabulary to compute the output probability of each target word
        """
        def sampled_loss(logits, labels):
            labels = tf.reshape(labels, [-1, 1])    # flatten and transpose
            return tf.nn.sampled_softmax_loss(weights=tf.transpose(w),
                                              biases=b,
                                              inputs=logits,
                                              labels=labels,
                                              num_sampled=config.NUM_SAMPLES,
                                              num_classes=config.DEC_VOCAB)
        self.softmax_loss_function = sampled_loss

        single_cell = tf.contrib.rnn.GRUCell(config.HIDDEN_SIZE)
        self.cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(config.NUM_LAYERS)])

    def _create_loss(self):
        print("Creating loss... \nIt might take a couple of minutes depending on how many buckets you have")
        start = time.time()
        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):

            # Info on embedding_attention_seq2seq
            """ embedding_attention_seq2seq:

                Returns:
                    tf.contrib.legacy_seq2seq.embedding_attention_seq2seq returns a tuple of
                    the form (outputs, state), where:

                      outputs: A list of the same length as decoder_inputs of 2D Tensors with
                        shape [batch_size x num_decoder_symbols] containing the generated
                        outputs.
                        # For first try this would be: 63 x config.DEC_VOCAB

                      state: The state of each decoder cell at the final time-step.
                        It is a 2D Tensor of shape [batch_size x cell.state_size]
                        # 63 x config.HIDDEN_SIZE
            """

            # Info on setattr()
            """ Example 1: How setattr() works in Python
            
            class Person:
                name = 'Adam'
            p = Person()
            print('Before modification:', p.name)
            # setting name to 'John'
            setattr(p, 'name', 'John')
            print('After modification:', p.name)
            """
            setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
            setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

            # Info on do_decode and output_projection
            """ do_decode:-
            
            feed_previous: Boolean or scalar Boolean Tensor; 
            - if True, only the first of decoder_inputs will be used (the "<GO>" symbol), and all other decoder
            inputs will be taken from previous outputs (as in embedding_rnn_decoder).
            - if False, decoder_inputs are used as given (the standard decoder case).
            
                output_projection: 
            output_projection: None or a pair (W, B) of output projection weights and biases; 
            W has shape [output_size x num_decoder_symbols] and B has shape [num_decoder_symbols];
            # output_size = config.HIDDEN_SIZE
            - if provided and feed_previous=True, each fed previous output will first be multiplied by W and added B.
            # see below (line 142)
            """
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, self.cell,
                num_encoder_symbols=config.ENC_VOCAB,
                num_decoder_symbols=config.DEC_VOCAB,
                embedding_size=config.HIDDEN_SIZE,      # -no of neurons
                output_projection=self.output_projection,
                feed_previous=do_decode)

        # Info on model_with_buckets
        """ model_with_buckets:
        
         Returns:
            A tuple of the form (outputs, losses), where:
              outputs: The outputs for each bucket. Its j'th element consists of a list of 2D Tensors.  The shape of 
                output tensors can be either [batch_size x output_size] or [batch_size x num_decoder_symbols]
                depending on the seq2seq model used.
              
              losses: List of scalar Tensors, representing losses for each bucket, or,
                if per_example_loss is set, a list of 1D batch-sized float Tensors.
        """
        # Testing
        if self.fw_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoder_inputs,
                                        self.decoder_inputs,
                                        self.targets,
                                        self.decoder_masks,     # weights
                                        config.BUCKETS,
                                        seq2seq = lambda x, y: _seq2seq_f(x, y, True),  # *
                                        softmax_loss_function = self.softmax_loss_function)

            # * seq2seq: A sequence-to-sequence model function; it takes 2 input that agree with encoder_inputs and
            # decoder_inputs, and returns a pair consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
            # do_decode - if True, only the first of decoder_inputs will be used (the "<GO>" symbol), and all other
            #               decoder inputs will be taken from previous outputs (as in embedding_rnn_decoder)

            # If we use an output projection, we need to project outputs for decoding
            if self.output_projection:
                for bucket in range(len(config.BUCKETS)):       # weight                        # bias
                    self.outputs[bucket] = [tf.matmul(output, self.output_projection[0]) + self.output_projection[1]
                                            for output in self.outputs[bucket]]

        # Training
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoder_inputs,
                                        self.decoder_inputs,
                                        self.targets,
                                        self.decoder_masks,
                                        config.BUCKETS,
                                        seq2seq = lambda x, y: _seq2seq_f(x, y, False), # *
                                        softmax_loss_function=self.softmax_loss_function)
            # do_decode - if False, decoder_inputs are used as given (the standard decoder case).
        print('Time:', time.time() - start)

    def _create_optimizer(self):
        print('Create optimizer... \nIt might take a couple of minutes depending on how many buckets you have')
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            if not self.fw_only:    # i.e if back prop is done  # i.e training
                self.optimizer = tf.train.GradientDescentOptimizer(config.LR)
                trainables = tf.trainable_variables()   # Returns all variables created with `trainable=True`
                self.gradient_norms = []
                self.train_ops = []
                start = time.time()
                for bucket in range(len(config.BUCKETS)):

                    # Info on gradients
                    """ tf.clip_by_global_norm:
                    
                    Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`, this operation 
                    returns a list of clipped tensors `list_clipped` and the global norm (`global_norm`) of all 
                    tensors in `t_list`.
                    
                    - t_list: tf.gradients(self.losses[bucket], trainables)
                    - clip_norm: config.MAX_GRAD_NORM
                    -------------------------------------------------------------------------------------------
                        tf.gradients:
                    
                    Constructs symbolic derivatives of sum of `ys` w.r.t. x in `xs`.
                    - ys = self.losses[bucket]
                    - xs = trainables
                     
                    `ys` and `xs` are each a `Tensor` or a list of tensors.  `grad_ys` is a list of `Tensor`, holding 
                    the gradients received by the `ys`. The list must be the same length as `ys`.
                    
                    `gradients()` adds ops to the graph to output the derivatives of `ys` with respect to `xs`. It 
                    returns a list of `Tensor` of length `len(xs)` where each tensor is the `sum(dy/dx)` for y in `ys`.
                    
                    --------------------------------------------------------------------------------------------
                        tf.apply_gradients:
                        
                    Apply gradients to variables.
                    This is the second part of `minimize()`. It returns an `Operation` that applies gradients.
                
                    Args:
                      grads_and_vars: List of (gradient, variable) pairs as returned by `compute_gradients()`.
                      global_step: Optional `Variable` to increment by one after the variables have been updated.
                      
                    Returns:
                      An `Operation` that applies the specified gradients. If `global_step` was not None, 
                      that operation also increments `global_step`.
                    """
                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket], trainables),
                                                                 config.MAX_GRAD_NORM)
                    self.gradient_norms.append(norm)

                    # Perform gradient update:
                    self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables),
                                                                         global_step=self.global_step))
                    print('Creating optimizer for bucket {} took {} seconds'.format(bucket, time.time() - start))
                    start = time.time()


    def _create_summary(self):
        pass

    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()

