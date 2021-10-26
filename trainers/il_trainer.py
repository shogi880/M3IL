import tensorflow as tf
import time
from tensorflow.python import debug as tf_debug
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


VAL_SUMMARY_INTERVAL = 1000
SUMMARY_INTERVAL = 1000
SAVE_INTERVAL = 25000
EVAL_INTERVAL = 25000
EVAL_TRAINENV_INTERVAL = 25000


class ILTrainer(object):

    def __init__(self, pipeline, outputs, generator, iterations,
                 summary_writer=None, eval=None, eval_trainenv=None, checkpoint_iter=-1):
        self.pipeline = pipeline
        self.generator = generator
        self.outputs = outputs
        self.iterations = iterations
        self.summary_writer = summary_writer
        self.eval = eval
        self.eval_trainenv = eval_trainenv
        self.checkpoint_iter = checkpoint_iter

        if eval is not None:
            # Convenience for plotting eval successes in tensorboard
            self.eval_summary_in = tf.placeholder(tf.float32)
            self.eval_summary = tf.summary.scalar('evaluation_success',
                                                  self.eval_summary_in)
            self.new_eval_summary_in = tf.placeholder(tf.float32)
            self.new_eval_summary = tf.summary.scalar('new_evaluation_success',
                                                      self.new_eval_summary_in)
        if eval_trainenv is not None:
            # Convenience for plotting eval successes in tensorboard
            self.eval_trainenv_summary_in = tf.placeholder(tf.float32)
            self.eval_trainenv_summary = tf.summary.scalar('evaluation_trainenv_success',
                                                           self.eval_trainenv_summary_in)

    def train(self, eval_only=False):
        sess = self.pipeline.get_session()
#        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        train_handle, validation_handle = self.generator.get_handles(sess)

        outputs = self.outputs
        total_loss = self.pipeline.get_loss()
        print(total_loss)
        train_op = self.pipeline.get_train_op(total_loss)
        train_summaries = self.pipeline.get_summaries('train')
        validation_summaries = self.pipeline.get_summaries('validation')

        tf.global_variables_initializer().run()

        # Load if we have supplied a checkpoint

        if not eval_only:
            resume_itr = self.pipeline.load()
            print('Setup Complete. Starting training...')
        
            for itr in range(resume_itr, self.iterations + 1):
                
                # skip training if evaluation only

                fetches = [train_op]

                feed_dict = {
                    self.generator.handle: train_handle,
                    outputs['training']: True
                }

                if itr % SUMMARY_INTERVAL == 0 or itr < 200:
                    fetches.append(total_loss)
                    if self.summary_writer is not None:
                        fetches.append(train_summaries)

                start = time.time()
                result = sess.run(fetches, feed_dict)

                if itr % SUMMARY_INTERVAL == 0 or itr < 200:
                    print('Summary iter', itr, '| Loss:',
                        result[1], '| Time:', time.time() - start)
                    if self.summary_writer is not None:
                        self.summary_writer.add_summary(sess, result[-1], itr)

                if (itr % VAL_SUMMARY_INTERVAL == 0 and self.summary_writer is not None):
                    feed_dict = {
                        self.generator.handle: validation_handle,
                        outputs['training']: False
                    }
                    result = sess.run([validation_summaries], feed_dict)
                    self.summary_writer.add_summary(sess, result[0], itr)

                if itr % EVAL_INTERVAL == 0 and itr > 1 and self.eval is not None:
                    acc, new_acc = self.eval.evaluate(itr)
                    print('Evaluation on old test setat iter %d. Success rate: %.2f (test)' % (itr, acc))
                    print('Evaluation on new test set at iter %d. Success rate: %.2f (test)' % (itr, new_acc))
                    if self.summary_writer is not None:
                        eval_success = sess.run(
                            self.eval_summary, {self.eval_summary_in: acc})
                        self.summary_writer.add_summary(sess, eval_success, itr)
                    
                        new_eval_success = sess.run(
                            self.new_eval_summary, {self.new_eval_summary_in: new_acc})
                        self.summary_writer.add_summary(sess, new_eval_success, itr)

                if itr % EVAL_TRAINENV_INTERVAL == 0 and itr > 1 and self.eval_trainenv is not None:
                    acc = self.eval_trainenv.evaluate(itr)
                    print('Evaluation on train set at iter %d. Success rate: %.2f (train)' % (itr, acc))
                    if self.summary_writer is not None:
                        eval_trainenv_success = sess.run(
                            self.eval_trainenv_summary, {self.eval_trainenv_summary_in: acc})
                        self.summary_writer.add_summary(sess, eval_trainenv_success, itr)

                if itr % SAVE_INTERVAL == 0:
                    self.pipeline.save(itr)
        elif self.checkpoint_iter > -1:
            itr = self.checkpoint_iter
            print(f'Evaling with {itr} checkpoint.')
            self.pipeline.load(itr)
            # import pdb; pdb.set_trace()
            if itr % EVAL_INTERVAL == 0 and itr > 1 and self.eval is not None:
                acc, new_acc = self.eval.evaluate(itr)
                print('Evaluation on old test setat iter %d. Success rate: %.2f (test)' % (itr, acc))
                print('Evaluation on new test set at iter %d. Success rate: %.2f (test)' % (itr, new_acc))
                if self.summary_writer is not None:
                    eval_success = sess.run(
                        self.eval_summary, {self.eval_summary_in: acc})
                    self.summary_writer.add_summary(sess, eval_success, itr)
                
                    new_eval_success = sess.run(
                        self.new_eval_summary, {self.new_eval_summary_in: new_acc})
                    self.summary_writer.add_summary(sess, new_eval_success, itr)

            if itr % EVAL_TRAINENV_INTERVAL == 0 and itr > 1 and self.eval_trainenv is not None:
                acc = self.eval_trainenv.evaluate(itr)
                print('Evaluation on train set at iter %d. Success rate: %.2f (train)' % (itr, acc))
                if self.summary_writer is not None:
                    eval_trainenv_success = sess.run(
                        self.eval_trainenv_summary, {self.eval_trainenv_summary_in: acc})
                    self.summary_writer.add_summary(sess, eval_trainenv_success, itr)
        else:
            for itr in range(25000, 400001, 25000):
                print(f'Evaling with {itr} checkpoint.')
                self.pipeline.load(itr)
                # import pdb; pdb.set_trace()
                if itr % EVAL_INTERVAL == 0 and itr > 1 and self.eval is not None:
                    acc, new_acc = self.eval.evaluate(itr)
                    print('Evaluation on old test setat iter %d. Success rate: %.2f (test)' % (itr, acc))
                    print('Evaluation on new test set at iter %d. Success rate: %.2f (test)' % (itr, new_acc))
                    if self.summary_writer is not None:
                        eval_success = sess.run(
                            self.eval_summary, {self.eval_summary_in: acc})
                        self.summary_writer.add_summary(sess, eval_success, itr)
                    
                        new_eval_success = sess.run(
                            self.new_eval_summary, {self.new_eval_summary_in: new_acc})
                        self.summary_writer.add_summary(sess, new_eval_success, itr)

                if itr % EVAL_TRAINENV_INTERVAL == 0 and itr > 1 and self.eval_trainenv is not None:
                    acc = self.eval_trainenv.evaluate(itr)
                    print('Evaluation on train set at iter %d. Success rate: %.2f (train)' % (itr, acc))
                    if self.summary_writer is not None:
                        eval_trainenv_success = sess.run(
                            self.eval_trainenv_summary, {self.eval_trainenv_summary_in: acc})
                        self.summary_writer.add_summary(sess, eval_trainenv_success, itr)
                # finish if evaluation only
                # if eval_only:
                #     break
