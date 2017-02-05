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

def extract_features(args, data_paths):
    # load model
    network = importlib.import_module(args.model_def, 'inference')
    with tf.Graph().as_default():
        print('Building inference graph')
        data_labels = tf.zeros(tf.shape(data_paths), dtype=tf.int32)
        eval_image_batch, eval_label_batch = facenet.read_and_augument_data(
            data_paths,
            data_labels,
            args.image_size,
            args.batch_size,
            1,
            False,
            False,
            False,
            args.nrof_preprocess_threads,
            shuffle=False,
            file_ext=args.file_ext)
        # Node for input images
        eval_image_batch.set_shape((None, args.image_size, args.image_size, 3))
        eval_image_batch = tf.identity(eval_image_batch, name='input')
        eval_prelogits, _ = network.inference(eval_image_batch, 1.0,
            phase_train=False, weight_decay=0.0, reuse=False)
        eval_embeddings = tf.nn.l2_normalize(eval_prelogits, 1, 1e-10, name='embeddings')

    # forward
        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners(sess=sess)

        # Create a saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
        pretrained_model = os.path.expanduser(args.pretrained_model)
        pretrained_model = tf.train.latest_checkpoint(pretrained_model)
        print('Restoring pretrained model: %s' % pretrained_model)
        saver.restore(sess, pretrained_model)

        embedding_features = []
        with sess.as_default():
            print("Ready to test.")
            nrof_images = data_labels.get_shape().as_list()[0]
            import math
            nrof_batches = int(math.ceil(nrof_images / float(args.batch_size)))
            print("num_batches", nrof_batches)
            for i in range(nrof_batches):
                t = time.time()
                embedding_feature_batch = sess.run([eval_embeddings])
                if i == 0:
                    embedding_features = embedding_feature_batch[0]
                else:
                    embedding_features = np.concatenate((embedding_features, embedding_feature_batch[0]), 0)
                print('Batch %d in %.3f seconds' % (i, time.time()-t))
            print("Test finishs.")

    print("debug", embedding_features)
    return embedding_features

def save_features(args, data_paths, feature_batch):
    import matio

    feature_dir = args.feature_dir
    data_dir = args.data_dir

    for data_path, feature in zip(data_paths, feature_batch):
        rel_path = os.path.relpath(data_path, data_dir)
        rel_path = rel_path[:-3] + "bin" # jpg->bin
        feature_path = os.path.join(feature_dir, rel_path)

        tf.gfile.MakeDirs(os.path.dirname(feature_path))
        matio.save_mat(feature_path, feature)



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
    parser.add_argument('--feature_dir', type=str,
        help='Path to the feature directory containing inferred face embedding features.',
        default='~/datasets/FaceScrub/features/')
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
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augumentation) threads.', default=4)
    parser.add_argument('--file_ext', type=str,
        help='The file extension for the dataset.', default='jpg', choices=['jpg', 'png'])
 
    return parser.parse_args(argv)
  
def get_data_paths(data_dir):
    # prepare data
    data_dir = args.data_dir
    print("input data dir is", data_dir)

    data_paths = []
    for abs_root, dirs, files in os.walk(data_dir):
        print("searching", abs_root)
        for filename in files:
            ext = filename.split(".")[-1]
            if ext != 'jpg' and ext != 'png':
                continue
            data_path = os.path.join(abs_root, filename)
            data_paths.append(data_path)
            #debug
            print("add data_path", data_path)
        #TODO: remove this line
        #if abs_root == "/root/datasets/zackhsiao/FaceScrub/test_cropped/facescrub_aligned/":
        #    continue
        #break
    return data_paths

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    data_paths = get_data_paths(args.data_dir)
    feature_batch = extract_features(args, data_paths)
    save_features(args, data_paths, feature_batch)
