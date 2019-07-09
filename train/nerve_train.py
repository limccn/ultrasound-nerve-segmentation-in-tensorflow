
import os.path
import time

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
import model.nerve_net as nerve_net
from utils.experiment_manager import make_checkpoint_path

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # 使用第二块GPU（从0开始）
os.environ["CUDA_VISIBLE_DEVICES"] = "1"       # 使用第二块GPU（从0开始）


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('base_dir', '../checkpoints',
                            """dir to store trained net """)
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """ training batch size """)
tf.app.flags.DEFINE_integer('max_steps', 2501000,
                            """ max number of steps to train """)
tf.app.flags.DEFINE_float('keep_prob', 0.668,
                            """ keep probability for dropout """)
tf.app.flags.DEFINE_float('learning_rate', 1e-5,
                            """ keep probability for dropout """)


TRAIN_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)
print(TRAIN_DIR)

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    image, mask = nerve_net.inputs(FLAGS.batch_size) 
    # create and unrap network
    prediction = nerve_net.inference(image, FLAGS.keep_prob) 
    # calc error
    error = nerve_net.loss_image(prediction, mask) 
    # train hopefuly 
    train_op = nerve_net.train(error, FLAGS.learning_rate)
    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())   
    #for i, variable in enumerate(variables):
    #  print '----------------------------------------------'
    #  print variable.name[:variable.name.index(':')]

    # Summary op
    summary_op = tf.summary.merge_all()
    
 
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()
    
    #cpu
    #sess_config=tf.ConfigProto(
    #    device_count={"CPU":1},
    #    inter_op_parallelism_threads=1,
    #    intra_op_parallelism_threads=1,
    #)

    #gpu
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth=True
    sess_config.gpu_options.per_process_gpu_memory_fraction=0.9

    # Start running operations on the Graph.
    #sess = tf.Session()
    sess=tf.Session(config=sess_config) 

    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)

    # Start que runner
    tf.train.start_queue_runners(sess=sess)

    # Summary op
    #graph_def = sess.graph.as_graph_def(add_shapes=True)
    graph_def = sess.graph_def
    summary_writer = tf.summary.FileWriter(TRAIN_DIR, sess.graph)

    for step in range(FLAGS.max_steps+1):
      t = time.time()
      _ , loss_value = sess.run([train_op, error],feed_dict={})
      elapsed = time.time() - t

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step%100 == 0:
        summary_str = sess.run(summary_op, feed_dict={})
        summary_writer.add_summary(summary_str, step) 
        print("loss value at " + str(loss_value))
        print("time per batch is " + str(elapsed))

      if step%1000 == 0:
        checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + TRAIN_DIR)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(TRAIN_DIR):
    tf.gfile.DeleteRecursively(TRAIN_DIR)
  tf.gfile.MakeDirs(TRAIN_DIR)
  train()

if __name__ == '__main__':
  tf.app.run()
