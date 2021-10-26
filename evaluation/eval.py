import os
from data import utils
import numpy as np
import imageio


class Eval(object):

    def __init__(self, sess, dataset, outputs, supports, num_tasks,
                 num_trials, log_dir=".", record_gifs=False, render=True, env_type='test'):
        self.time_horizon = dataset.time_horizon
        self.sess = sess
        self.env_type = env_type
        if env_type == 'test':
            self.demos = dataset.test_set()
            self.new_demos = dataset.new_test_set()
        elif env_type == 'train':
            self.demos = dataset.training_set()[0]
        elif env_type == 'valid':
            self.demos = dataset.training_set()[1]
        else:
            raise RuntimeError('Unrecognised env_type.')
        self.supports = supports
        self.num_tasks = num_tasks
        self.num_trials = num_trials
        self.log_dir = log_dir
        self.record_gifs = record_gifs
        self.render = render
        self.record_gifs_dir = os.path.join(self.log_dir, 'evaluated_gifs')
        self.outputs = outputs

    def evaluate(self, iter):
        raise NotImplementedError("Override this function.")

    def get_embedding(self, task_index, demo_indexes):
        image_files = [
            self.demos[task_index][j]['image_files'] for j in demo_indexes]
        states = [
            self.demos[task_index][j]['states'] for j in demo_indexes]
        outs = [
            self.demos[task_index][j]['actions'] for j in demo_indexes] 
        language = [
            self.demos[task_index][j]['language'] for j in demo_indexes]
        
#        print(len(language))

        feed_dict = {
            self.outputs['input_image_files']: image_files,
            self.outputs['input_states']: states,
            self.outputs['input_outputs']: outs,
            self.outputs['input_language']: language,
        }
       # import pdb;pdb.set_trace()
        embedding, = self.sess.run(self.outputs['sentences'], feed_dict=feed_dict)
        return embedding
    
    def new_get_embedding(self, task_index, demo_indexes):
        image_files = [
            self.new_demos[task_index][j]['image_files'] for j in demo_indexes]
        states = [
            self.new_demos[task_index][j]['states'] for j in demo_indexes]
        outs = [
            self.new_demos[task_index][j]['actions'] for j in demo_indexes] 
        language = [
            self.new_demos[task_index][j]['language'] for j in demo_indexes]
        
#        print(len(language))

        feed_dict = {
            self.outputs['input_image_files']: image_files,
            self.outputs['input_states']: states,
            self.outputs['input_outputs']: outs,
            self.outputs['input_language']: language,
        }
       # import pdb;pdb.set_trace()
        embedding, = self.sess.run(
            self.outputs['sentences'], feed_dict=feed_dict)
        return embedding
    
    def get_action(self, obs, state, embedding):
        feed_dict = {
            self.outputs['ctrnet_images']: [[obs]],
            self.outputs['ctrnet_states']: [[state]],
            self.outputs['sentences']: [embedding],
        }
        action, = self.sess.run(
            self.outputs['output_actions'], feed_dict=feed_dict)
        return action

    def create_gif_dir(self, iteration_dir, task_id):
        gifs_dir = None
        if self.record_gifs:
            gifs_dir = os.path.join(iteration_dir, 'task_%d' % task_id)
            utils.create_dir(gifs_dir)
        return gifs_dir

    def save_gifs(self, observations, gifs_dir, trial):
        if self.record_gifs:
            video = np.array(observations)
            record_gif_path = os.path.join(
                gifs_dir, 'cond%d.samp0.gif' % trial)
            imageio.mimwrite(record_gif_path, video)
