from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import time
import sys
import tensorflow as tf
import numpy as np
import importlib
import argparse
import tensorflow.contrib.slim as slim
import facenet


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def extract_features(args):

    # prepare data
    data_dir = args.data_dir
    data_paths = []
    for abs_root, dirs, files in os.walk(data_dir):
        for filename in files:
            ext = filename.split(".")[-1]
            if ext != 'jpg':
                continue
            data_paths += os.path.join(abs_root, filename)
            break
        break
    #debug
    print("data_paths=")
    print(data_paths)

    # load model
    network = importlib.import_module(args.model_def, 'inference')
    with tf.Graph().as_default():
        print('Building inference graph')
        label_list = tf.zeros(tf.shape(data_paths))
        eval_image_batch, eval_label_batch = facenet.read_and_augument_data(
            data_paths,
            [],
            args.image_size,
            1,
            None,
            False,
            False,
            False,
            args.nrof_preprocess_threads,
            shuffle=False)
        # Node for input images
        eval_image_batch.set_shape((None, args.image_size, args.image_size, 3))
        eval_image_batch = tf.identity(eval_image_batch, name='input')
        eval_prelogits, _ = network.inference(eval_image_batch, 1.0,
            phase_train=False, weight_decay=0.0, reuse=True)
        eval_embeddings = tf.nn.l2_normalize(eval_prelogits, 1, 1e-10, name='embeddings')

    # forward
        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        #sess.run(tf.global_variables_initializer())
        #sess.run(tf.local_variables_initializer())
        #tf.train.start_queue_runners(sess=sess)

        embedding_features = sess.run([eval_embeddings], feed_dict={inputs: eval_image_batch})
        #with sess.as_default():
        #    # Evaluate
        #    evaluate(sess, eval_embeddings, eval_label_batch, actual_issame, args.lfw_batch_size, args.seed, 
        #        args.lfw_nrof_folds, log_dir, step)

    return embedding_features

def evaluate(sess, embeddings, labels, actual_issame, batch_size, 
        seed, nrof_folds, log_dir, step):
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on input images')
    embedding_size = embeddings.get_shape()[1] # [0] is batch size
    nrof_images = embeddings.get_shape()[0]
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    for i in range(nrof_batches):
        t = time.time()
        emb, lab = sess.run([embeddings, labels])
        emb_array[lab] = emb
        print('Batch %d in %.3f seconds' % (i, time.time()-t))
        
    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, seed, actual_issame, nrof_folds=nrof_folds)
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.', required=True)
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='~/datasets/facescrub/fs_aligned:~/datasets/casia/casia-webface-aligned')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.nn4')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=96)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--decov_loss_factor', type=float,
        help='DeCov loss factor.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--log_histograms', 
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
 
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    extract_features(parse_arguments(sys.argv[1:]))
