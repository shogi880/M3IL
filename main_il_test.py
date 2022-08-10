from pickle import FALSE
from tensorflow.python.platform import flags
from data.mil_sim_reach import MilSimReach
from data.mil_sim_push import MilSimPush
import datetime
from consumers.control import Control
from consumers.eval_consumer import EvalConsumer
from consumers.generator_consumer import GeneratorConsumer
from consumers.imitation_loss import ImitationLoss
from consumers.margin_loss import MarginLoss
from consumers.reparam_loss import ReparamLoss
from consumers.task_embedding import TaskEmbedding
from data.data_sequencer import DataSequencer
from data.generator import Generator
from networks.cnn import CNN
from trainers.il_trainer import ILTrainer
from trainers.pipeline import Pipeline
from trainers.summary_writer import SummaryWriter
from trains import Task
import os
from networks.save_load import Saver, Loader
import numpy as np
import tensorflow as tf

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# Dataset/method options
flags.DEFINE_string(
    'dataset', 'sim_push', 'One of sim_reach, sim_push.')
flags.DEFINE_string(
    'datasetdir', os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '../../datasets'),
    'The directory where the dataset exists.')

# Training Options
flags.DEFINE_integer(
    'iterations', 400000, 'The number of training iterations.')
flags.DEFINE_integer(
    'batch_size', 64, 'The number of tasks sampled per batch (aka batch size).')
flags.DEFINE_float(
    'lr', 0.0001, 'The learning rate.')
flags.DEFINE_integer(
    'support', 1, 'The number of support examples per task (aka k-shot).')
flags.DEFINE_integer(
    'query', 1, 'The number of query examples per task.')
flags.DEFINE_integer(
    'embedding', 20, 'The embedding size.')
flags.DEFINE_float(
    'grad_clip', 10.0, 'Gradient clipping (Default: None).')

# Model Options
flags.DEFINE_string(
    'activation', 'relu', 'One of relu, elu, or leaky_relu.')
flags.DEFINE_bool(
    'max_pool', False, 'Use max pool rather than strides.')
flags.DEFINE_list(
    'filters', [16, 16, 16, 16], 'List of filters per convolution layer.')
flags.DEFINE_list(
    'kernels', [5, 5, 5, 5], 'List of kernel sizes per convolution layer.')
flags.DEFINE_list(
    'strides', [2, 2, 2, 2], 'List of strides per convolution layer. '
    'Can be None if using max pooling.')
flags.DEFINE_list(
    'fc_layers', [200, 200, 200], 'List of fully connected nodes per layer.')
flags.DEFINE_float(
    'drop_rate', 0.0, 'Dropout probability. 0 for no dropout.')
flags.DEFINE_string(
    'norm', "layer", 'One of layer, batch, or None')

# Loss Options
flags.DEFINE_float(
    'lambda_embedding', 0.0000001, 'Lambda for the embedding loss.')
flags.DEFINE_float(
    'lambda_support', 1.0, 'Lambda for the support control loss.')
flags.DEFINE_float(
    'lambda_query', 1.0, 'Lambda for the query control loss.')
flags.DEFINE_string(
    'loss_type', 'reparam_loss', 'One of margin_loss or reparam_loss.')
flags.DEFINE_bool(
    'clip_sigma', True, 'If we should clip scale when using reparam loss')
flags.DEFINE_bool(
    'use_prior', True, 'If we should use standard Normal as Prior (Default: False).')
flags.DEFINE_bool(
    'support_sampling', True, 'If we should sample from support embedding (Default: True).')
flags.DEFINE_string(
    'uncert_type', 'logsigma', 'Interpretation of 2nd output of embedding network  (Default: logsigma).')
flags.DEFINE_string(
    'kl_type', 'support', 'Choice of prior (p) of kl term [support, standard] (Default: support).')
flags.DEFINE_float(
    'margin', 0.1, 'The margin for the embedding loss.')

# Logging, Saving, and Eval Options
flags.DEFINE_bool(
    'summaries', True, 'If false do not write summaries (for tensorboard).')
flags.DEFINE_bool(
    'save', True, 'If false do not save network weights.')
flags.DEFINE_bool(
    'load', False, 'If we should load a checkpoint.')
flags.DEFINE_string(
    'logdir', './log_1', 'The directory to store summaries and checkpoints.')
flags.DEFINE_bool(
    'eval', False, 'If evaluation should be done.')
flags.DEFINE_bool(
    'eval_trainenv', False, 'If evaluation in training environment should be done.')
flags.DEFINE_bool(
    'eval_only', False, 'Evaluate only and not resume training.')
flags.DEFINE_list(
    'eval_num_tasks', [2, 2], 'Number of tasks for evaluation in test and training env (default: [74, 20]).')
flags.DEFINE_integer(
    'eval_num_trials', 6, 'Number of trials for evaluation (default: 6).')
flags.DEFINE_integer(
    'checkpoint_iter', -1, 'The checkpoint iteration to restore '
                           '(-1 for latest model).')
flags.DEFINE_string(
    'checkpoint_dir', None, 'The checkpoint directory.')
flags.DEFINE_bool(
    'no_mujoco', True, 'Run without Mujoco. Eval should be False.')
flags.DEFINE_string(
    'image_type', 'first_last', 'Choose image_type, [first, first_last, last, first_last_diff]')

# Additional options for logging
flags.DEFINE_string(
    'experiment_name', 'experiment', 'Experiment name.')

flags.DEFINE_string(
    'sentence_type', 'both', '[language, image, both]')
# Additional options for language_instructions
flags.DEFINE_string(
    'embedding_type', "image", 'choose from [image_embedding, language_embedding, both_embedding, id_embedding]')
flags.DEFINE_string(
    'sentence', 'sentence', 'choose which instruction from [sentence, word, 15types, 30types] to using. default is sentence')
flags.DEFINE_integer(
    'random_seed', 1, 'random_seed: 1~5')
flags.DEFINE_string(
    'reparam_embedding_type', "None", 'from [None, kl, kl_2image, kl_2image]')
flags.DEFINE_string(
    'fusion_type', "add", 'Fusion type of image and language embedding. Choose from [add, concat]')
flags.DEFINE_bool(
    'task_id', False, 'Using task_id instead of instruction if you want (Default:False)')
FLAGS = flags.FLAGS

seed = FLAGS.random_seed
np.random.seed(seed)
tf.set_random_seed(seed)

if not FLAGS.no_mujoco:
    from evaluation.eval_mil_push import EvalMilPush

# naming the experiment_name by the type of network .

# when you eval, please add random_seed to the experiment name.
if not FLAGS.eval_only:
    FLAGS.experiment_name += "_" + str(FLAGS.random_seed)

if FLAGS.lambda_embedding not in [0, 1]:
    FLAGS.experiment_name += "_" + str(FLAGS.lambda_embedding)

if FLAGS.reparam_embedding_type != 'None':
    FLAGS.experiment_name += "_" + FLAGS.reparam_embedding_type

if FLAGS.fusion_type != 'add':
    FLAGS.experiment_name += "_" + FLAGS.fusion_type

if FLAGS.image_type in ['first', 'last']:
    FLAGS.experiment_name += "_" + FLAGS.image_type

print("Started at ", datetime.datetime.now().strftime('%Y-%m-%d-%H'))
print("Experiment_name: ", FLAGS.experiment_name)
# from delogger.presets.profiler import logger

# @logger.line_memory_profile

# if FLAGS.eval_only:
#     _task = Task.init(project_name='tecnets_tf', task_name=FLAGS.experiment_name, task_type='testing')
# else:
#     _task = Task.init(project_name='tecnets_tf', task_name=FLAGS.experiment_name)

filters = list(map(int, FLAGS.filters))
kernels = list(map(int, FLAGS.kernels))
strides = list(map(int, FLAGS.strides))
fc_layers = list(map(int, FLAGS.fc_layers))

data = None
if FLAGS.dataset == 'sim_reach':
    data = MilSimReach(datasetdir=FLAGS.datasetdir)
elif FLAGS.dataset == 'sim_push':
    data = MilSimPush(datasetdir=FLAGS.datasetdir, sentence=FLAGS.sentence)
else:
    raise RuntimeError('Unrecognised dataset.')

loader = saver = None
savedir = os.path.join(FLAGS.logdir, FLAGS.experiment_name, 'model')
# if FLAGS.save and os.path.exists(os.path.join(FLAGS.logdir, FLAGS.experiment_name)):
#     raise RuntimeError('The experiment already exists.')
if FLAGS.save:
    saver = Saver(savedir=savedir)


def main():
    loader = None

    if FLAGS.image_type in ['first', 'test_first']:
        sequencer = DataSequencer('first', data.time_horizon)
    elif FLAGS.image_type == 'first_last_diff':
        sequencer = DataSequencer('first_last_diff', data.time_horizon)
    elif FLAGS.image_type in ['last', 'test_last']:
        sequencer = DataSequencer('last', data.time_horizon)
    else:
        sequencer = DataSequencer('first_last', data.time_horizon)
    net = CNN(filters=filters,
              fc_layers=fc_layers,
              kernel_sizes=kernels,
              strides=strides,
              max_pool=FLAGS.max_pool,
              drop_rate=FLAGS.drop_rate,
              norm=FLAGS.norm,
              activation=FLAGS.activation,
              fusion_type=FLAGS.fusion_type,
              embedding_size=2 * FLAGS.embedding)
    gen = Generator(dataset=data,
                    batch_size=FLAGS.batch_size,
                    support_size=FLAGS.support,
                    query_size=FLAGS.query,
                    data_sequencer=sequencer,
                    sentence=FLAGS.sentence)
    # sequencer = DataSequencer('first_last_diff', data.time_horizon)
    # sequencer = DataSequencer('first_plus_first_last_diff', data.time_horizon)

    generator_consumer = GeneratorConsumer(
        gen, data, FLAGS.support, FLAGS.query)
    if FLAGS.loss_type == 'margin_loss':
        task_emb = TaskEmbedding(network=net,
                                 embedding_size=FLAGS.embedding,
                                 include_state=False,
                                 include_action=False)
        ml = MarginLoss(margin=FLAGS.margin,
                        loss_lambda=FLAGS.lambda_embedding)
    elif FLAGS.loss_type == 'reparam_loss':
        # Reparam version of Task embedding and its loss
        task_emb = TaskEmbedding(network=net,
                                 embedding_size=2 * FLAGS.embedding,
                                 include_state=False,
                                 include_action=False,
                                 embedding_type=FLAGS.embedding_type,
                                 sentence=FLAGS.sentence)
        ml = ReparamLoss(embedding_size=FLAGS.embedding,
                         loss_lambda=FLAGS.lambda_embedding,
                         clip_sigma=FLAGS.clip_sigma,
                         use_prior=FLAGS.use_prior,
                         support_sampling=FLAGS.support_sampling,
                         uncert_type=FLAGS.uncert_type,
                         sentence_type=FLAGS.sentence_type,
                         embedding_type=FLAGS.embedding_type,
                         reparam_embedding_type=FLAGS.reparam_embedding_type)
    else:
        raise RuntimeError('Unrecognised loss type.')
    ctr = Control(network=net,
                  action_size=data.action_size,
                  include_state=True)
    il = ImitationLoss(support_lambda=FLAGS.lambda_support,
                       query_lambda=FLAGS.lambda_query)

    consumers = [generator_consumer, task_emb, ml, ctr, il]

    if FLAGS.load:
        checkpointdir = os.path.join(
            FLAGS.checkpoint_dir, FLAGS.experiment_name, 'model')
        loader = Loader(savedir=checkpointdir,
                        checkpoint=FLAGS.checkpoint_iter)
    p = Pipeline(consumers, saver=saver, loader=loader,
                 learning_rate=FLAGS.lr, grad_clip=FLAGS.grad_clip)
    train_outs = p.get_outputs()
    # print('========================')
    summary_w = None
    if FLAGS.eval_only:
        log_dir = os.path.join(FLAGS.logdir, FLAGS.experiment_name, "eval",
                               str(FLAGS.image_type) + "_sentence_type_" +
                               str(FLAGS.sentence_type),
                               "random_seed_" + str(FLAGS.random_seed) + "_" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
    else:
        log_dir = os.path.join(
            FLAGS.logdir, FLAGS.experiment_name, "sentence_type_" + str(FLAGS.sentence_type))
    print(f"""
          {'*' * 70} 
          Testing {FLAGS.experiment_name},
          with random_seed: {FLAGS.random_seed}
          logdir...: {log_dir} 
          {'*' * 70}
          """)
    if FLAGS.summaries:
        summary_w = SummaryWriter(log_dir)

    eval = None
    eval_trainenv = None
    if FLAGS.eval:
        disk_images = FLAGS.dataset != 'sim_to_real_place'
        econs = EvalConsumer(data, sequencer, FLAGS.support,
                             disk_images, FLAGS.sentence)
        if FLAGS.loss_type == 'margin_loss':
            task_emb = TaskEmbedding(network=net,
                                     embedding_size=FLAGS.embedding,
                                     include_state=False,
                                     include_action=False)
            ml = MarginLoss(margin=FLAGS.margin,
                            loss_lambda=FLAGS.lambda_embedding)
        elif FLAGS.loss_type == 'reparam_loss':
            # Reparam version of Task embedding and its loss
            task_emb = TaskEmbedding(network=net,
                                     embedding_size=2 * FLAGS.embedding,
                                     include_state=False,
                                     include_action=False,
                                     embedding_type=FLAGS.embedding_type,
                                     sentence=FLAGS.sentence)
            ml = ReparamLoss(embedding_size=FLAGS.embedding,
                             loss_lambda=FLAGS.lambda_embedding,
                             sentence_type=FLAGS.sentence_type,
                             embedding_type=FLAGS.embedding_type,
                             reparam_embedding_type=FLAGS.reparam_embedding_type)
        else:
            raise RuntimeError('Unrecognised loss type.')
        ctr = Control(network=net,
                      action_size=data.action_size,
                      include_state=True)
        peval = Pipeline([econs, task_emb, ml, ctr])
        outs = peval.get_outputs()

        num_tasks = list(map(int, FLAGS.eval_num_tasks))
        eval = EvalMilPush(sess=p.get_session(),
                           dataset=data,
                           outputs=outs,
                           supports=FLAGS.support,
                           num_tasks=num_tasks[0],
                           num_trials=FLAGS.eval_num_trials,
                           log_dir=log_dir,
                           record_gifs=False,
                           #    render=True,
                           render=False,
                           env_type='test')
        if FLAGS.eval_trainenv:
            eval_trainenv = EvalMilPush(sess=p.get_session(),
                                        dataset=data,
                                        outputs=outs,
                                        supports=FLAGS.support,
                                        num_tasks=num_tasks[1],
                                        num_trials=FLAGS.eval_num_trials,
                                        log_dir=log_dir,
                                        record_gifs=False,
                                        render=False,
                                        env_type='train')
    trainer = ILTrainer(pipeline=p,
                        outputs=train_outs,
                        generator=gen,
                        iterations=FLAGS.iterations,
                        summary_writer=summary_w,
                        eval=eval,
                        eval_trainenv=eval_trainenv,
                        checkpoint_iter=FLAGS.checkpoint_iter)

    trainer.train(eval_only=FLAGS.eval_only)


main()
