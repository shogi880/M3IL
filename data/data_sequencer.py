from data import utils
import tensorflow as tf
# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior()
import random

VALID_SEQUENCE_STRATEGIES = ['first', 'last', 'first_last', 'all', 'first_last_diff', 'first_plus_first_last_diff']


class DataSequencer(object):

    def __init__(self, sequence_strategy, time_horizon):
        self.sequence_strategy = sequence_strategy
        self.time_horizon = time_horizon
        if sequence_strategy not in VALID_SEQUENCE_STRATEGIES:
            raise ValueError('%s is not a valid sequence embedding strategy.'
                             % sequence_strategy)

        self.frames = 2
        if sequence_strategy == 'all':
            self.frames = self.time_horizon

    def load(self, images, states, outputs, instructions):
        is_image_file = images.dtype == tf.string
        # Embedding images
        if self.sequence_strategy == 'first':
            if is_image_file:
                loaded_images = [utils.tf_load_image(images, 0),
                                 utils.tf_load_image(images, 0)]
            else:
                loaded_images = [images[0], images[0]]
            emb_states = [states[0], states[0]]
            emb_outputs = [outputs[0], outputs[0]]
            emb_instructions = instructions
        elif self.sequence_strategy == 'first_last_diff':
            if is_image_file:
                loaded_images = [utils.tf_load_image(images,self.time_horizon - 1) - utils.tf_load_image(images, 0), utils.tf_load_image(images,self.time_horizon - 1) - utils.tf_load_image(images, 0)]
            else:
                loaded_images = [images[-1] - images[0], images[0] - images[-1]]
            emb_states = [states[0], states[0]]
            emb_outputs = [outputs[0], states[0]]
            emb_instructions = instructions
        elif self.sequence_strategy == 'last':
            if is_image_file:
                loaded_images = [utils.tf_load_image(images, self.time_horizon - 1), utils.tf_load_image(images, self.time_horizon - 1)]
            else:
                loaded_images = [images[self.time_horizon - 1], images[self.time_horizon - 1]]
            emb_states = [states[-1], states[-1]]
            emb_outputs = [outputs[-1], outputs[-1]]
            emb_instructions = instructions
        elif self.sequence_strategy == 'first_last':
            if is_image_file:
                loaded_images = [utils.tf_load_image(images, 0),
                                 utils.tf_load_image(images,
                                                     self.time_horizon - 1)]
            else:
                loaded_images = [images[0], images[self.time_horizon - 1]]
            emb_states = [states[0], states[-1]]
            emb_outputs = [outputs[0], outputs[-1]]
            emb_instructions = instructions
            # print(instructions.shape)
        elif self.sequence_strategy == 'all':
            if is_image_file:
                loaded_images = [utils.tf_load_image(images, t)
                          for t in range(self.time_horizon)]
            else:
                loaded_images = images
            emb_states = [states[t]
                          for t in range(self.time_horizon)]
            emb_outputs = [outputs[t]
                           for t in range(self.time_horizon)]
        else:
            raise ValueError(
                '%s is not a valid sequence embedding strategy.'
                % self.sequence_strategy)
        return loaded_images, emb_states, emb_outputs, emb_instructions
