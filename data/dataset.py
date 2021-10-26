import os
import pickle
from natsort import natsorted
import numpy as np
import random


class Dataset(object):

    def __init__(self, name, img_shape, state_size, action_size, time_horizon,
                 training_size=None, validation_size=None, datasetdir=None, sentence='sentence', task_id=False):
        self.name = name
        self.img_shape = img_shape
        self.state_size = state_size
        self.action_size = action_size
        self.time_horizon = time_horizon
        self.training_size = training_size
        self.validation_size = validation_size
        self.sentence = sentence
        self.task_id = task_id
        self.data_root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), datasetdir, name)

    def training_set(self):
        tasks = self.load('train', self.training_size + self.validation_size)
        return tasks[:self.training_size], tasks[-self.validation_size:]

    def test_set(self):
        return self.load("test")
    
    def new_test_set(self):
        return self.load("new_test")

    def load(self, train_or_test, count=None):
        """Expected to be the test or train folder"""
        print("loading the ", train_or_test, "files.")
        train_test_dir = os.path.join(self.data_root, train_or_test)
        tasks = []
        for task_f in natsorted(os.listdir(train_test_dir)):

            task_path = os.path.join(train_test_dir, task_f)
            if not os.path.isdir(task_path):
                continue
            pkl_file = task_path + '.pkl'
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            example_img_folders = natsorted(os.listdir(task_path))
            examples = []
            for e_idx, ex_file in enumerate(example_img_folders):
                img_path = os.path.join(task_path, ex_file)
                example = {
                    'image_files': img_path,
                    'actions': data['actions'][e_idx],
                    'states': data['states'][e_idx]
                }
                if 'demo_selection' in data:
                    example['demo_selection'] = data['demo_selection']
                    npy_name = data["demo_selection"].split('/')[-1].split('.')[0]
                    if self.sentence == 'word':
                        # print('Using 768_push_the_N_to_the_red_area language instrcution...')
                        language_npy = f'/root/share/TecNets/datasets/768_word_instruction/{npy_name}.npy'
                    elif self.sentence == '15types':
                        language_npy = f'/root/share/TecNets/datasets/768_15_types_instruction/{npy_name}.npy'
                    elif self.sentence == '30types':
                        language_npy = f'/root/share/TecNets/datasets/768_30_types_instruction/{npy_name}.npy'
                    else:
                        language_npy = f'/root/share/TecNets/datasets/768_push_the_N_to_the_red_area/{npy_name}.npy'
                        
                    # example['language'] = [np.load(language_npy), np.load(id_npy) / 1102.0]
                    language = np.load(language_npy)
                    if self.task_id:
                        id_npy = f'/root/share/TecNets/datasets/768_push_the_N_to_the_red_area/id_{npy_name}.npy'
                        # language = np.full_like(language, np.load(id_npy) / 1102.)
                        language = np.full_like(language, np.load(id_npy) / 1.)
                    example['language'] = language
                examples.append(example)
            tasks.append(examples)
            if count is not None and len(tasks) >= count:
                break
        return tasks

    def get_outputs(self):
        return {
            'actions': (self.action_size,)
        }

    def get_inputs(self):
        return {
            'states': (self.state_size,)
        }
