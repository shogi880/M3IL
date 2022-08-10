import json
import os
import numpy as np
import random
from gym.envs.mujoco.pusher import PusherEnv
from evaluation.eval import Eval
from data import utils


XML_FOLDER = "/root/workspace/gym/gym/envs/mujoco/assets"
# XML_FOLDER = "/root/share/M3IL/gym/gym/envs/mujoco/assets"


class EvalMilPush(Eval):

    def _load_env(self, xml, new_test=False):
        """[summary]

        Args:
            xml ([type]): [description]
            new_test (bool, optional): [description]. Defaults to False.

        Raises:
            RuntimeError: [description]

        Returns:
            [type]: [description]
        """
        xml = xml[xml.rfind('pusher'):]
        if new_test:
            xml_file = 'sim_push_xmls/train_ensure_woodtable_distractor_%s' % xml
        elif self.env_type == 'test':
            xml_file = 'sim_push_xmls/test2_ensure_woodtable_distractor_%s' % xml
        elif self.env_type == 'train':
            xml_file = 'sim_push_xmls/train_ensure_woodtable_distractor_%s' % xml
        else:
            raise RuntimeError('Cannot read xml (unrecognised env_type).')

        # import ipdb; ipdb.set_trace()
        xml_file = os.path.join(XML_FOLDER, xml_file)
        env = PusherEnv(**{'xml_file': xml_file, 'distractors': True})
        # env.set_visibility(self.render)
        env.render('rgb_array')
        viewer = env.viewer
        viewer.autoscale()
        viewer.cam.trackbodyid = -1
        viewer.cam.lookat[0] = 0.4
        viewer.cam.lookat[1] = -0.1
        viewer.cam.lookat[2] = 0.0
        viewer.cam.distance = 0.75
        viewer.cam.elevation = -50
        viewer.cam.azimuth = -90
        return env, xml

    def _eval_success(self, obs):
        obs = np.array(obs)
        target = obs[:, -3:-1]
        obj = obs[:, -6:-4]
        dists = np.sum((target - obj) ** 2, 1)  # distances at each time step
        return np.sum(dists < 0.017) >= 10

    def evaluate(self, iter):
        """[summary]

        Args:
            iter ([type]): [description]

        Raises:
            RuntimeError: [description]
        """
        print("Evaluating at iteration: %i" % iter)
        if self.env_type == 'test':
            iter_dir = os.path.join(self.record_gifs_dir, 'iter_%i' % iter)
            # new_iter_dir = os.path.join(self.record_gifs_dir, 'new_iter_%i' % iter)
            # utils.create_dir(iter_dir)
        elif self.env_type == 'train':
            iter_dir = os.path.join(
                self.record_gifs_dir,
                'iter_%i_train' %
                iter)
        else:
            raise RuntimeError('Cannot create dir (unrecognised env_type).')
        utils.create_dir(iter_dir)

        results = {}
        successes = []

        for i in range(self.num_tasks):
            ith_successes = []

            # demo_selection will be an xml file
            env, xml = self._load_env(self.demos[i][0]['demo_selection'])
            selected_demo_indexs = random.sample(
                range(len(self.demos[i])), self.supports)

            embedding = self.get_embedding(i, selected_demo_indexs)
            # gifs_dir = self.create_gif_dir(iter_dir, i)
            for j in range(self.num_trials):
                env.reset()
                observations = []
                world_state = []
                for t in range(self.time_horizon):
                    env.render('rgb_array')
                    # Observation is shape  (100,100,3)
                    obs, state = env.get_current_image_obs()
                    # print(obs)
                    obs = ((obs / 255.0) * 2.) - 1.
                    action = self.get_action(obs, state, embedding)
                    ob, reward, done, reward_dict = env.step(
                        np.squeeze(action))
                    world_state.append(np.squeeze(ob))
                    if done:
                        break

                if self._eval_success(world_state):
                    ith_successes.append(1.)
                else:
                    ith_successes.append(0.)
                # self.save_gifs(observations, gifs_dir, j)

            env.render(close=True)
            results[i] = {"xml": xml,
                          "demo_idx": selected_demo_indexs,
                          "suc": np.mean(ith_successes),
                          "trial": ith_successes}
            successes.append(ith_successes)
        final_suc = np.mean(successes)
        json_data = {"final_suc": final_suc, "tasks": results}
        with open(os.path.join(iter_dir, "result.json"), 'w') as f:
            json.dump(json_data, f)

        # if self.env_type != 'train':
        #     results = {}
        #     successes = []
        #     for i in range(self.num_tasks):
        #         ith_successes = []

        #         # demo_selection will be an xml file
        #         env, xml = self._load_env(
        #             self.new_demos[i][0]['demo_selection'], new_test=True)
        #         selected_demo_indexs = random.sample(
        #             range(len(self.new_demos[i])), self.supports)

        #         embedding = self.new_get_embedding(i, selected_demo_indexs)
        #         # gifs_dir = self.create_gif_dir(new_iter_dir, i)
        #         for j in range(self.num_trials):
        #             env.reset()
        #             observations = []
        #             world_state = []
        #             for t in range(self.time_horizon):
        #                 env.render()
        #                 # Observation is shape  (100,100,3)
        #                 obs, state = env.get_current_image_obs()
        #                 observations.append(obs)
        #                 obs = ((obs / 255.0) * 2.) - 1.
        #                 action = self.get_action(
        #                     obs, state, embedding)
        #                 ob, reward, done, reward_dict = env.step(
        #                     np.squeeze(action))
        #                 action = self.get_action(obs, state, embedding)  # instructionを入れずに
        #                 ob, reward, done, reward_dict = env.step(np.squeeze(action))
        #                 world_state.append(np.squeeze(ob))
        #                 if done:
        #                     break

        #             if self._eval_success(world_state):
        #                 ith_successes.append(1.)
        #             else:
        #                 ith_successes.append(0.)
        #             # self.save_gifs(observations, gifs_dir, j)

        #         env.render(close=True)
        #         results[i] = {"xml": xml,
        #                     "demo_idx": selected_demo_indexs,
        #                     "suc": np.mean(ith_successes),
        #                     "trial": ith_successes}
        #         successes.append(ith_successes)
        #     new_final_suc = np.mean(successes)
        #     json_data = {"new_final_suc": new_final_suc, "tasks": results}
        #     with open(os.path.join(new_iter_dir, "result.json"), 'w') as f:
        #         json.dump(json_data, f)

        # print("Final success rate is %.5f (%s)" % (final_suc, self.env_type))
        # if self.env_type != 'train':
            # print("Final success rate is %.5f new (%s)" % (new_final_suc, self.env_type))
            # return final_suc, new_final_suc
        # else:
        return final_suc
