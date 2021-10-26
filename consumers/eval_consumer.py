from consumers.consumer import Consumer
import tensorflow as tf
# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior()
from data import utils


class EvalConsumer(Consumer):

    def __init__(self, dataset, data_sequencer, support, disk_images=True, sentence='sentence'):
        self.dataset = dataset
        self.data_sequencer = data_sequencer
        self.support = support
        self.disk_images = disk_images
        self.sentence = sentence
        super().__init__()

    def consume(self, inputs):

        if self.disk_images:
            # (Examples,)
            input_image = tf.placeholder(tf.string, (self.support,))
        else:
            # (Examples, timesteps)
            input_image = tf.placeholder(tf.float32,
                                         (None, None) + self.dataset.img_shape)
        input_states = tf.placeholder(
            tf.float32,
            (self.support, self.dataset.time_horizon, self.dataset.state_size))
        input_outputs = tf.placeholder(
            tf.float32,
            (self.support, self.dataset.time_horizon, self.dataset.action_size))
        # input_language = tf.placeholder(
                    # tf.float32, (self.support, 1, 768))
        if self.sentence in ['sentence', 'word']:
            input_language = tf.placeholder(tf.float32, (self.support, 1, 768))
        else:
            print('placeholder: ', int(self.sentence[:2]))
            input_language = tf.placeholder(tf.float32, (self.support, int(self.sentence[:2]), 768))
        # (B. W, H, C)
        input_ctr_image = tf.placeholder(tf.float32,
                                         (None, 1) + self.dataset.img_shape)
        input_ctr_state = tf.placeholder(tf.float32,
                                         (None, 1, self.dataset.state_size))

        training = tf.placeholder_with_default(False, None)

        stacked_embnet_images, bs, cs, ins = [], [], [], []
        for i in range(self.support):
            embnet_images, embnet_states, embnet_outputs, embnet_language = (
                self.data_sequencer.load(
                    input_image[i], input_states[i], input_outputs[i], input_language[i]))
            embnet_images = utils.preprocess(embnet_images)
            stacked_embnet_images.append(embnet_images)
            bs.append(embnet_states)
            cs.append(embnet_outputs)
            ins.append(embnet_language)


        embnet_images = tf.stack(stacked_embnet_images)
        embnet_images = tf.expand_dims(embnet_images, axis=0)  # set batchsize 1

        embnet_states = tf.stack(bs)
        embnet_states = tf.expand_dims(embnet_states, axis=0)

        embnet_outputs = tf.stack(cs)
        embnet_outputs = tf.expand_dims(embnet_outputs, axis=0)

        embnet_language = tf.stack(ins)
        embnet_language = tf.expand_dims(embnet_language, axis=0)

        embnet_images.set_shape(
            (None, None, self.data_sequencer.frames) + self.dataset.img_shape)
        embnet_states.set_shape(
            (None, None, self.data_sequencer.frames, self.dataset.state_size))
        embnet_outputs.set_shape(
            (None, None, self.data_sequencer.frames, self.dataset.action_size))
        # embnet_language.set_shape(
            # (None, None, self.data_sequencer.frames, 768))
        if self.sentence in ['sentence', 'word']:
            embnet_language.set_shape((None, None, 1, 768))
        else:
            embnet_language.set_shape(
                (None, None, int(self.sentence[:2]), 768))

        return {
            'embnet_images': embnet_images,
            'embnet_states': embnet_states,
            'embnet_outputs': embnet_outputs,
            'embnet_language': embnet_language,
            'input_image_files': input_image,
            'input_states': input_states,
            'input_outputs': input_outputs,
            'input_language': input_language,
            'ctrnet_images': input_ctr_image,
            'ctrnet_states': input_ctr_state,
            'training': training,
            'support': tf.placeholder_with_default(self.support, None),
            'query': tf.placeholder_with_default(0, None),
        }
