import tensorflow as tf
# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior()
from networks.utils import *
from networks.input_output import *


class CNN(object):

    def __init__(
            self,
            filters,
            fc_layers,
            kernel_sizes,
            strides=None,
            max_pool=False,
            drop_rate=0.0,
            norm=None,
            activation='relu',
            fusion_type='add',
            embedding_size=20):
        """Initializes a standard CNN network.

        :param filters: List of number of filters per convolution layer.
        :param fc_layers: List of fully connected units per layer.
        :param normalization: String defining the type of normalization.
        """
        self.filters = filters
        self.fc_layers = fc_layers
        self.norm = norm
        self.drop_rate = drop_rate
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.max_pool = max_pool
        self.fusion_type = fusion_type
        self.strides = strides if strides is not None else [1] * len(fc_layers)
        self.embedding_size=embedding_size
        if not max_pool and strides is None:
            raise RuntimeError('No dimensionality reduction.')

    def _pre_layer_util(self, layer, cur_layer_num, ins):
        for cin in ins:
            if cin.layer_num > cur_layer_num:
                break
            elif cin.layer_num == cur_layer_num:
                if cin.merge_mode == 'concat':
                    layer = tf.concat([layer, cin.tensor], axis=cin.axis)
                elif cin.merge_mode == 'addition':
                    layer += cin.tensor
                elif cin.merge_mode == 'multiply':
                    layer *= cin.tensor
                else:
                    raise RuntimeError('Unrecognised merging method for %s.' %
                                       cin.name)
        return layer

    def _post_layer_util(self, layer, training, norm):

        if self.drop_rate > 0:
            layer = tf.layers.dropout(layer, rate=0.5, training=training)

        act_fn = activation(self.activation)
        if norm and self.norm is not None:
            if self.norm == 'batch':
                layer = tf.contrib.layers.batch_norm(
                    layer, is_training=training, activation_fn=act_fn)
            elif self.norm == 'layer':
                layer = tf.contrib.layers.layer_norm(
                    layer, activation_fn=act_fn)
            else:
                raise RuntimeError('Unsupported normalization method: %s'
                                   % self.norm)
        else:
            layer = act_fn(layer)
        return layer

    def forward(self, inputs, heads, training):
        """Inputs want to be fused in at different times. """
        inputs = sorted(inputs, key=lambda item: item.layer_num)
        conv_inputs = list(filter(lambda item: item.layer_type == 'conv', inputs))
        fc_inputs = list(filter(lambda item: item.layer_type == 'fc', inputs))
        outputs = {}

        if heads[0].name in ["both_embedding", "language_embedding"]:
            language_layer = fc_inputs[0].tensor
            del fc_inputs[0]

        if heads[0].name != "language_embedding":
            if conv_inputs[0].layer_num > 0:
                raise RuntimeError('Need an input tensor.')
            elif len(conv_inputs) > 1 and conv_inputs[1].layer_num == 0:
                raise RuntimeError('Can only have one main input tensor.')

            layer = conv_inputs[0].tensor
            del conv_inputs[0]

            # import pdb; pdb.set_trace()
            for i, (filters, ksize, stride) in enumerate(zip(self.filters, self.kernel_sizes, self.strides)):
                layer = self._pre_layer_util(layer, i, conv_inputs)
                layer = tf.layers.conv2d(layer, filters, ksize, stride, 'same')
                layer = self._post_layer_util(layer, training, True)
            layer = tf.layers.flatten(layer)

            for i, fc_layers in enumerate(self.fc_layers):
                layer = self._pre_layer_util(layer, i, fc_inputs)
                layer = tf.layers.dense(layer, fc_layers)
                layer = self._post_layer_util(layer, training, False)
        
        if heads[0].name in ["both_embedding", "language_embedding"]:
            # import pdb; pdb.set_trace()
            # print(language_layer)

            # language_layer = tf.keras.layers.Reshape((1, 1, 768))(language_layer)
            language_layer = tf.layers.dense(language_layer, 1024)
            language_layer = tf.layers.dense(language_layer, 512)
            language_layer = tf.layers.dense(language_layer, 256)
            language_layer = tf.layers.dense(language_layer, 128)
            language_layer = tf.layers.dense(language_layer, 64)

        if heads[0].name == "both_embedding":
            image_layer = tf.layers.dense(layer, 64)
            outputs['language_embedding'] = activation(self.activation)(tf.layers.dense(language_layer, self.embedding_size))
            outputs['image_embedding'] = activation(self.activation)(tf.layers.dense(image_layer, self.embedding_size))        
        
            if self.fusion_type == 'concat':
                # import pdb; pdb.set_trace()
                layer = tf.keras.layers.Concatenate(-1)([image_layer, language_layer])
            else:
                layer = tf.keras.layers.Add()([image_layer, language_layer])

        act_fn = activation(heads[0].activation)

        if heads[0].name == "language_embedding":
            output = tf.layers.dense(language_layer, heads[0].nodes)
        else:
            output = tf.layers.dense(layer, heads[0].nodes)
        outputs[heads[0].name] = output if act_fn is None else act_fn(output)

        return outputs    