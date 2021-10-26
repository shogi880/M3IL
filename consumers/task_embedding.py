from consumers.consumer import Consumer
from networks.input_output import NetworkHead, NetworkInput
import tensorflow as tf
import numpy as np
# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior()
VALID_FRAME_COLLAPSE = ['concat']

class TaskEmbedding(Consumer):

    def __init__(
            self,
            network,
            embedding_size,
            frame_collapse_method='concat',
            include_state=False,
            include_action=False,
            embedding_type="image_embedding",
            only_fist_image=False,
            kl=False,
            sentence='sentence'):
        self.network = network
        self.embedding_size = embedding_size
        self.include_state = include_state
        self.include_action = include_action
        # embedding_type = [image_embedding, language_embedding, both_embedding]
        self.embedding_type = embedding_type
        
        if frame_collapse_method not in VALID_FRAME_COLLAPSE:
            raise ValueError('%s is not a valid frame collapse method.'
                             % frame_collapse_method)
        self.frame_collapse_method = frame_collapse_method
        self.only_fist_image = only_fist_image
        self.sentence = sentence

    def _squash_input(self, tensor, shape, batch_size, support_plus_query):
        return tf.reshape(tensor, tf.concat(
            [[batch_size * support_plus_query], shape[2:]], axis=0))

    def _expand_output(self, tensor, batch_size, support_plus_query):
        return tf.reshape(tensor, (batch_size, support_plus_query, -1))

    def consume(self, inputs):
        net_ins = []
        embed_images = self.get(inputs, 'embnet_images')
        support = self.get(inputs, 'support')
        query = self.get(inputs, 'query')
        if self.frame_collapse_method == 'concat':
            embed_images = tf.concat(tf.unstack(embed_images, axis=2), axis=-1)

        embed_images_shape = tf.shape(embed_images)
        batch_size = embed_images_shape[0]
        support_plus_query = embed_images_shape[1]
        assertion_op = tf.assert_equal(
            support_plus_query, support + query,
            message='Support and Query size is different than expected.')
#         print(embed_images.shape) # 1, 5, 125, 125, 6
        with tf.control_dependencies([assertion_op]):
            # Condense to shape (batch_size*(support_plus_query),w,h,c)
            reshaped_images = self._squash_input(
                embed_images, embed_images_shape, batch_size,
                support_plus_query)
        # print(reshaped_images.shape)
        net_ins.append(NetworkInput(name='embed_images', layer_type='conv',
                                layer_num=0, tensor=reshaped_images, merge_mode='concat'))

        # if self.include_language:
        if self.embedding_type != "image_embedding":
            # import pdb; pdb.set_trace()
            embnet_language = self.get(inputs, 'embnet_language')
            # if self.frame_collapse_method == 'concat':
                # embnet_language = tf.concat(tf.unstack(embnet_language, axis=2), axis=-1)
            # import pdb; pdb.set_trace()
            if self.sentence == ['15types', '30types']:
                random_num = np.random.randint(embnet_language.shape[-2])
            else:
                random_num = 0
            embnet_language = embnet_language[:, :, random_num]
            reshaped_instruction = self._squash_input(embnet_language, tf.shape(embnet_language), batch_size,
                support_plus_query)
            # print(reshaped_instruction.shape)
            net_ins.append(NetworkInput(name='embnet_language', layer_type='fc',
                layer_num=0, tensor=reshaped_instruction, merge_mode='concat'))

        net_out = NetworkHead(name=self.embedding_type,
                              nodes=self.embedding_size)
        # print(self.embedding_type)
        with tf.variable_scope('task_embedding_net', reuse=tf.AUTO_REUSE):
            outputs = self.network.forward(net_ins, [net_out],
                                           self.get(inputs, 'training'))

        # Convert to (Batch, support_query, emb_size)
        embedding = tf.reshape(self.get(outputs, self.embedding_type),
                    (batch_size, support_plus_query, self.embedding_size))

        outputs['support_embedding'] = embedding[:, :support]
        outputs['query_embedding'] = embedding[:, support:]


# =========================== kl実装==================================
        if self.embedding_type == "both_embedding":
            image_embedding = tf.reshape(self.get(outputs,'image_embedding'),
                (batch_size, support_plus_query, self.embedding_size))

            outputs['support_image_embedding'] = image_embedding[:, :support]

            language_embedding = tf.reshape(self.get(outputs,'language_embedding'),
                (batch_size,support_plus_query,self.embedding_size))

            outputs['support_language_embedding'] = language_embedding[:, :support]

# =========================== kl実装==================================
        # control_imageを
        inputs.update(outputs)
        return inputs
