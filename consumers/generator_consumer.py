from consumers.consumer import Consumer
import tensorflow as tf
# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior()

class GeneratorConsumer(Consumer):

    def __init__(self, generator, dataset, support, query):
        super().__init__()
        self.generator = generator
        self.dataset = dataset
        self.support = support
        self.query = query
        self.embnet_images = None
        self.ctrnet_images = None

    def consume(self, inputs):
        (embnet_images, embnet_states, embnet_outputs, embnet_language, ctrnet_images,
         ctrnet_states, ctrnet_outputs) = self.generator.next_element
        training = tf.placeholder(tf.bool)
        
#         print("embnet_language: ",embnet_language)
        
        embnet_images.set_shape((None, None, None) + self.dataset.img_shape)
        embnet_states.set_shape((None, None, None, self.dataset.state_size))
        embnet_outputs.set_shape((None, None, None, self.dataset.action_size))
        embnet_language.set_shape((None, None, None, 768))
        ctrnet_images.set_shape((None, 2,) + self.dataset.img_shape)
        ctrnet_states.set_shape((None, 2, self.dataset.state_size))
        ctrnet_outputs.set_shape((None, 2, self.dataset.action_size))

        self.embnet_images = embnet_images
        self.ctrnet_images = ctrnet_images
        print(embnet_language.shape)
#         print(embnet_language.shape, embnet_outputs.shape, embnet_states.shape, embnet_images.shape)
        return {
            'embnet_images': embnet_images,
            'embnet_states': embnet_states,
            'embnet_outputs': embnet_outputs,
            'embnet_language': embnet_language,
            'ctrnet_images': ctrnet_images,
            'ctrnet_states': ctrnet_states,
            'ctrnet_outputs': ctrnet_outputs,
            'training': training,
            'support': tf.placeholder_with_default(self.support, None),
            'query': tf.placeholder_with_default(self.query, None),
        }

    def get_summaries(self, prefix):
        # Grab the last frame for each task.
        # We know there should be at least 2 examples.
        # pass 
        # embnet_example_1 = self.verify(self.embnet_images)[:, 0, -1]
        # embnet_example_2 = self.verify(self.embnet_images)[:, 1, -1]
        # ctrnet_support = self.verify(self.ctrnet_images)[:, 0]
        ctrnet_query = self.verify(self.ctrnet_images)[:, 1]
        return [
            # tf.summary.image(prefix + '_embnet_example_1', embnet_example_1),
            # tf.summary.image(prefix + '_embnet_example_2', embnet_example_2),
            # tf.summary.image(prefix + '_ctrnet_support', ctrnet_support),
            tf.summary.image(prefix + '_ctrnet_query', ctrnet_query),
        ]
