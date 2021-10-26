import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior()
import multiprocessing
import logging
from data import utils


class Generator(object):

    def __init__(self, dataset, batch_size, support_size, query_size,
                 data_sequencer, sentence='sentence'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.support_size = support_size
        self.query_size = query_size
        self.data_sequencer = data_sequencer
        self.support_query_size = support_size + query_size
        self.train, self.validation = self.dataset.training_set()
        self.sentence = sentence
        self._construct()

    def _create_generator(self, data):

        num_tasks = len(data)
        samples_to_take = np.minimum(self.batch_size, num_tasks)
        if samples_to_take != self.batch_size:
            logging.warning('Batch size was greater than number of tasks.')
        # import pdb; pdb.set_trace()
        def gen():

            while True:
                # add instruction = []
                states, outputs, image_files, language = [], [], [], []
                task_indices = np.random.choice(
                    num_tasks, samples_to_take, replace=False)
                for index in task_indices:
                    task = data[index]
                    num_examples_of_task = len(task)
                    if num_examples_of_task < self.support_query_size:
                        raise RuntimeError(
                            'Tried to sample %d support and query samples,'
                            'but there are only %d samples of this task.'
                            % (self.support_query_size, num_examples_of_task))
                    # [chose 10(self.support_query_size) from 12 random number ]
                    sample_indices = np.random.choice(
                        num_examples_of_task, self.support_query_size,
                        replace=False)
                    sampled_examples = [task[sample_index] for sample_index in
                                        sample_indices]
                    states.append([ex['states'] for ex in sampled_examples])
                    outputs.append([ex['actions'] for ex in sampled_examples])
                    image_files.append(
                        [ex['image_files'] for ex in sampled_examples])
#                     print(index,task,type(task), len(task))
                    # language[0] = instruction, language[1] = id
                    language.append([ex['language'] for ex in sampled_examples])
                states = np.array(states)  # <class 'numpy.ndarray'> 100 10 ?
                outputs = np.array(outputs)
                image_files = np.array(image_files)  # <class 'numpy.ndarray'> 100 10 69 => fileの長さ 
                language = np.array(language)
                # language[:][0] = instruction, language[:][1] = id
                yield image_files, states, outputs, language
        return gen

    def _load_from_disk(self, image_files, states, outputs, language):
        
        embnet_images, embnet_states, embnet_outputs, embnet_language = [], [], [], []
        for i in range(self.support_query_size):
            images, emb_states, emb_outputs, emb_language = self.data_sequencer.load(image_files[i], states[i], outputs[i], language[i])
            # images will be of shape (sequence, w, h, 3)

            embnet_images.append(images)
            embnet_states.append(emb_states)
            embnet_outputs.append(emb_outputs)
            embnet_language.append(emb_language)
#         import pdb; pdb.set_trace()   
        # embnet_langauge[:][0] = instruction, embnet_langauge[:][1] = id -> tensor.
        embed_images = tf.stack(embnet_images)
        embnet_states = tf.stack(embnet_states)
        embnet_outputs = tf.stack(embnet_outputs)
        embnet_language = tf.stack(embnet_language)

        embnet_states.set_shape(
            (self.support_query_size, self.data_sequencer.frames, None))
        embnet_outputs.set_shape(
            (self.support_query_size, self.data_sequencer.frames, None))
        # embnet_language.set_shape(
                    # (self.support_query_size, 2, None))
        # setting language shape
        if self.sentence in ['sentence', 'word']:
            embnet_language.set_shape(
                    (self.support_query_size, 1, None))
        else:
            embnet_language.set_shape(
                    (self.support_query_size, int(self.sentence[:2]), None))

        

        # Grab a random timestep in one of the support and query trajectories
        ctrnet_timestep = tf.random_uniform(
            (2,), 0, self.dataset.time_horizon, tf.int32)
        # The first should be a support and the last should be a query
        ctrnet_images = [
            utils.tf_load_image(image_files[0], ctrnet_timestep[0]),
            utils.tf_load_image(image_files[-1], ctrnet_timestep[1])
        ]
        ctrnet_states = [states[0][ctrnet_timestep[0]],
                         states[-1][ctrnet_timestep[1]]]
        ctrnet_outputs = [outputs[0][ctrnet_timestep[0]],
                          outputs[-1][ctrnet_timestep[1]]]

        ctrnet_images = tf.stack(ctrnet_images)
        ctrnet_states = tf.stack(ctrnet_states)
        ctrnet_outputs = tf.stack(ctrnet_outputs)

        embed_images = utils.preprocess(embed_images)
        ctrnet_images = utils.preprocess(ctrnet_images)

        # embnet_langauge[:][0] = instruction, embnet_langauge[:][1] = id -> tensor.
        return (embed_images, embnet_states, embnet_outputs, embnet_language,
                ctrnet_images, ctrnet_states, ctrnet_outputs)

    def _construct_dataset(self, data, prefetch):
        dataset = tf.data.Dataset.from_generator(
            self._create_generator(data), (tf.string, tf.float32, tf.float32, tf.float32))
        dataset = dataset.apply(tf.contrib.data.unbatch()).map(
            map_func=self._load_from_disk,
            num_parallel_calls=multiprocessing.cpu_count())
        dataset = dataset.batch(self.batch_size).prefetch(prefetch)
        return dataset

    def _construct(self):
        train_dataset = self._construct_dataset(self.train, 5)
        validation_dataset = self._construct_dataset(self.validation, 1)
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, train_dataset.output_types, train_dataset.output_shapes)

        self.next_element = iterator.get_next()
        self.train_iterator = train_dataset.make_one_shot_iterator()
        self.validation_iterator = validation_dataset.make_one_shot_iterator()
        self.handle = handle

    def get_handles(self, sess):
        training_handle = sess.run(self.train_iterator.string_handle())
        validation_handle = sess.run(self.validation_iterator.string_handle())
        return training_handle, validation_handle
