
import os.path
import time

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
import model.nerve_net as nerve_net
from utils.experiment_manager import make_checkpoint_path
import utils.metric as metc

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # 使用第二块GPU（从0开始）
os.environ["CUDA_VISIBLE_DEVICES"] = "1"       # 使用第二块GPU（从0开始）


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('base_dir', '../checkpoints',
                            """dir to store trained net """)
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """ training batch size """)
tf.app.flags.DEFINE_integer('max_steps', 227980,
                            """ max number of steps to train """)
#tf.app.flags.DEFINE_float('keep_prob', 0.69315, #ln2
tf.app.flags.DEFINE_float('keep_prob', 0.668, # gd
                            """ keep probability for dropout """)
tf.app.flags.DEFINE_float('learning_rate', 1e-5,
                            """ keep probability for dropout """)
#tf.app.flags.DEFINE_integer('bn_gamma', 1,
#                            """ BatchNorm gamma """)
#tf.app.flags.DEFINE_integer('bn_beta', 0,
#                            """ BatchNorm beta """)
#tf.app.flags.DEFINE_integer("epoch_num", 25,
#                            """Epoch to train [25]""")

#file_count:5635
#record_count:73255


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
    # meric 
    metric = nerve_net.metric_image(prediction,mask)
    #metric2 = nerve_net.metric_distance(prediction,mask)
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
    sess_config.gpu_options.per_process_gpu_memory_fraction=0.99

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
        
      if step%32 == 0:
        # meric distance
        np_predictions = prediction.eval(session=sess)
        np_masks= mask.eval(session=sess)
        asd_arr=np.array([])
        obj_asd_arr=np.array([])
        msd_arr=np.array([])
        hd_arr=np.array([])
        assd_arr=np.array([])
        for it in range(0,np_masks.shape[0]):
            prediction_i = np_predictions[it,:,:,:]
            mask_i = np_masks[it,:,:,:]
            prediction_i[:][prediction_i[:]<=.5]=0
            prediction_i[:][prediction_i[:]>.5] =1   
            mask_i[:][mask_i[:]<=.5]=0
            mask_i[:][mask_i[:]>.5] =1
            # sample except
            if 0 == np.count_nonzero(prediction_i) \
                or 0 == np.count_nonzero(mask_i):
                continue
            asd_i = metc.asd(prediction_i,mask_i)
            obj_asd_i = metc.obj_asd(prediction_i,mask_i)
            msd_i = metc.msd(prediction_i,mask_i)
            assd_i = metc.assd(prediction_i,mask_i)
            hd_i = metc.hd(prediction_i,mask_i)
            #print("asd_i:%s msd_i:%s"%(asd_i,msd_i))
            asd_arr = np.append(asd_arr,asd_i)
            obj_asd_arr = np.append(obj_asd_arr,obj_asd_i)
            msd_arr = np.append(msd_arr,msd_i)
            assd_arr = np.append(assd_arr,assd_i)
            hd_arr = np.append(hd_arr,hd_i)

        summt = tf.Summary()
        if asd_arr.size >0:
            asd = np.mean(asd_arr)
            summt.value.add(tag="asd", simple_value = asd)
        if obj_asd_arr.size >0:
            obj_asd = np.mean(obj_asd_arr)
            summt.value.add(tag="obj_asd", simple_value = obj_asd)
        if msd_arr.size >0:
            msd = np.max(msd_arr)
            summt.value.add(tag="msd", simple_value = msd)
        if assd_arr.size >0:
            assd = np.mean(assd_arr)
            summt.value.add(tag="assd", simple_value = assd)
        if hd_arr.size >0:
            hd = np.mean(hd_arr)
            summt.value.add(tag="hd", simple_value = hd) 
        summary_writer.add_summary(summt,step)

        #tf.summary.scalar('ASD', asd)
        #tf.summary.scalar('MSD', msd)

      if step%32 == 0:
        metric_list = sess.run([metric],feed_dict={})
        if metric_list and len(metric_list) > 0:
            metric_matrix = metric_list[0]
            tpr = int(metric_matrix["tpr"]*1000)
            summt = tf.Summary()
            summt.value.add(tag="roc", simple_value = tpr)
            
            fpr = int(metric_matrix["fpr"]*1000)
            summary_writer.add_summary(summt,fpr)
            #print('metrics:'+str(metric_matrix))


      #if step%100 == 0:
         
      if step%32 == 0:
        summary_str = sess.run(summary_op, feed_dict={})
        summary_writer.add_summary(summary_str, step)
        print("loss value at " + str(loss_value))
        print("time per batch is " + str(elapsed))

      #if step%1000 == 0:
      #  checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
      #  saver.save(sess, checkpoint_path, global_step=step)  
      #  print("saved to " + TRAIN_DIR)
      
      #epoch
      if step%(2213) == 2212:
        epoch=1+step//2213
        summary_str = sess.run(summary_op, feed_dict={})
        summary_writer.add_summary(summary_str, step)
        print("epoch=%d,loss=%s,steps=%d "%(epoch,str(loss_value),step))
        # save
        if epoch>50:
            checkpoint_path = os.path.join(TRAIN_DIR, 'model_epoch%d.ckpt'%epoch)
            saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(TRAIN_DIR):
    tf.gfile.DeleteRecursively(TRAIN_DIR)
  tf.gfile.MakeDirs(TRAIN_DIR)
  train()

if __name__ == '__main__':
  tf.app.run()
