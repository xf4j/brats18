import tensorflow as tf
import os, re, time
import numpy as np
from skimage.transform import resize, warp, AffineTransform
from skimage import measure
import h5py

from base_model import BaseModel

class UNet3D(BaseModel):
    def __init__(self, sess, checkpoint_dir, log_dir, training_subjects, testing_subjects, testing_during_training, 
                 model_config):
        
        self.sess = sess
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        self.training_subjects = training_subjects
        self.testing_subjects = testing_subjects
        
        self.testing_during_training = testing_during_training
        
        self.epoch = int(model_config['epoch'])
        self.features_root = int(model_config['features_root'])
        self.conv_size = int(model_config['conv_size'])
        self.layers = int(model_config['layers'])
        self.dropout = float(model_config['dropout_ratio'])
        self.loss_type = str(model_config['loss_type'])
        self.im_size = list(map(int, model_config['input_sizes'].split(',')))
        self.batch_size = int(model_config['batch_size'])
        self.use_prelu = model_config['use_prelu'] == 'True'
        self.use_conv_for_pooling = model_config['use_conv_for_pooling'] == 'True'
        self.test_nstride = int(model_config['test_nstride'])
        
        # For simplicity, these values are just hard-coded
        self.input_features = 4
        self.nclass = 4
        self.class_labels = [0, 1, 2, 4]
        
        self.class_weights = None
        
        self.build_model()
        
        self.saver = tf.train.Saver(list(set(tf.trainable_variables() + tf.get_collection_ref('bn_collections'))))
        
    def build_model(self):
        self.images = tf.placeholder(tf.float32, shape=[None, self.im_size[0], self.im_size[1], self.im_size[2],
                                                        self.input_features], name='images')
        self.labels = tf.placeholder(tf.float32, shape=[None, self.im_size[0], self.im_size[1], self.im_size[2],
                                                        self.nclass], name='labels')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_ratio')
        
        with tf.variable_scope('constants'):
            self.mean = tf.get_variable('mean', [self.input_features])
            self.std = tf.get_variable('std', [self.input_features])
        
        conv_size = self.conv_size
        layers = self.layers

        deconv_size = 2
        pool_stride_size = 2
        pool_kernel_size = 3 # Use a larger kernel
        
        # Encoding path
        connection_outputs = []
        for layer in range(layers):
            features = 2**layer * self.features_root
            if layer == 0:
                prev = self.images
            else:
                prev = pool
                
            conv = self.conv3d(prev, features, conv_size, use_prelu=self.use_prelu, is_training=self.is_training,
                               scope='encoding' + str(layer))
            connection_outputs.append(conv)
            if self.use_conv_for_pooling:
                pool = self.strided_conv(conv, features, conv_size, pool_stride_size, use_prelu=self.use_prelu,
                                         is_training=self.is_training, scope='convpool' + str(layer))
            else:
                pool = tf.nn.max_pool3d(conv, ksize=[1, pool_kernel_size, pool_kernel_size, pool_kernel_size, 1],
                                        strides=[1, pool_stride_size, pool_stride_size, pool_stride_size, 1],
                                        padding='SAME')
        
        bottom = self.conv3d(pool, 2**layers * self.features_root, conv_size, use_prelu=self.use_prelu,
                             is_training=self.is_training, scope='bottom')
        bottom = tf.nn.dropout(bottom, self.keep_prob)
        
        # Decoding path
        for layer in range(layers):
            conterpart_layer = layers - 1 - layer
            features = 2**conterpart_layer * self.features_root
            if layer == 0:
                prev = bottom
            else:
                prev = conv_decoding
            
            shape = prev.get_shape().as_list()
            deconv_output_shape = [tf.shape(prev)[0], shape[1] * deconv_size, shape[2] * deconv_size,
                                   shape[3] * deconv_size, features]
            deconv = self.deconv3d(prev, deconv_output_shape, deconv_size, use_prelu=self.use_prelu, 
                                   is_training=self.is_training, scope='decoding' + str(conterpart_layer))
            cc = self.crop_and_concat(connection_outputs[conterpart_layer], deconv)
            conv_decoding = self.conv3d(cc, features, conv_size, use_prelu=self.use_prelu, is_training=self.is_training,
                                        scope='decoding' + str(conterpart_layer))
        
        with tf.variable_scope('logits'):
            w = tf.get_variable('w', [1, 1, 1, conv_decoding.get_shape()[-1], self.nclass],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            logits = tf.nn.conv3d(conv_decoding, w, strides=[1, 1, 1, 1, 1], padding='SAME')
            b = tf.get_variable('b', [self.nclass], initializer=tf.constant_initializer(0.0))
            logits = tf.nn.bias_add(logits, b)
        
        self.probs = tf.nn.softmax(logits)
        self.predictions = tf.argmax(self.probs, 4)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.argmax(self.labels, 4)), tf.float32))
                                  
        flat_logits = tf.reshape(logits, [-1, self.nclass])
        flat_labels = tf.reshape(self.labels, [-1, self.nclass])
        
        if self.class_weights is not None:
            class_weights = tf.constant(np.asarray(self.class_weights, dtype=np.float32))
            weight_map = tf.reduce_max(tf.multiply(flat_labels, class_weights), axis=1)
            loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
            weighted_loss = tf.multiply(loss_map, weight_map)
            cross_entropy_loss = tf.reduce_mean(weighted_loss)
        else:
            cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                                        labels=flat_labels))
        eps = 1e-5
        dice_value = 0
        dice_loss = 0
        
        for i in range(1, self.nclass):
            slice_prob = tf.squeeze(tf.slice(self.probs, [0, 0, 0, 0, i], [-1, -1, -1, -1, 1]), axis=4)
            slice_pred = tf.cast(tf.equal(self.predictions, i), tf.float32)
            slice_label = tf.squeeze(tf.slice(self.labels, [0, 0, 0, 0, i], [-1, -1, -1, -1, 1]), axis=4)
            intersection_prob = eps + tf.reduce_sum(tf.multiply(slice_prob, slice_label), axis=[1, 2, 3])
            intersection_pred = eps + tf.reduce_sum(tf.multiply(slice_pred, slice_label), axis=[1, 2, 3])
            union_prob = 2.0 * eps + tf.reduce_sum(slice_prob, axis=[1, 2, 3]) + tf.reduce_sum(slice_label, axis=[1, 2, 3])
            union_pred = 2.0 * eps + tf.reduce_sum(slice_pred, axis=[1, 2, 3]) + tf.reduce_sum(slice_label, axis=[1, 2, 3])
            dice_loss += tf.reduce_mean(tf.div(intersection_prob, union_prob))
            dice_value += tf.reduce_mean(tf.div(intersection_pred, union_pred))
            
        dice_loss = 1 - dice_loss * 2.0 / (self.nclass - 1)
        self.dice = dice_value * 2.0 / (self.nclass - 1)
        
        if self.loss_type == 'cross_entropy':
            self.loss = cross_entropy_loss
        elif self.loss_type == 'dice':
            self.loss = dice_loss
        elif self.loss_type == 'both':
            self.loss = cross_entropy_loss + dice_loss
        else:
            raise ValueError("Unknown cost function: " + self.loss_type)
        
        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        self.dice_summary = tf.summary.scalar('dice', self.dice)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(self.loss)
        
    def conv3d(self, input_, output_dim, f_size, use_prelu, is_training, scope='conv3d'):
        with tf.variable_scope(scope):
            # VGG network uses two 3*3 conv layers to effectively increase receptive field
            w1 = tf.get_variable('w1', [f_size, f_size, f_size, input_.get_shape()[-1], output_dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1 = tf.nn.conv3d(input_, w1, strides=[1, 1, 1, 1, 1], padding='SAME')
            b1 = tf.get_variable('b1', [output_dim], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.bias_add(conv1, b1)
            bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_training, scope='bn1',
                                               variables_collections=['bn_collections'])
            if use_prelu:
                r1 = self.prelu(bn1, 'r1')
            else:
                r1 = tf.nn.relu(bn1)
            
            w2 = tf.get_variable('w2', [f_size, f_size, f_size, output_dim, output_dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2 = tf.nn.conv3d(r1, w2, strides=[1, 1, 1, 1, 1], padding='SAME')
            b2 = tf.get_variable('b2', [output_dim], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.bias_add(conv2, b2)
            bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_training, scope='bn2',
                                               variables_collections=['bn_collections'])
            if use_prelu:
                r2 = self.prelu(bn2, 'r2')
            else:
                r2 = tf.nn.relu(bn2)
            return r2
    
    def strided_conv(self, input_, output_dim, f_size, stride, use_prelu, is_training, scope='strided_conv'):
        with tf.variable_scope(scope):
            w = tf.get_variable('w1', [f_size, f_size, f_size, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv = tf.nn.conv3d(input_, w, strides=[1, stride, stride, stride, 1], padding='SAME')
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, b)
            bn = tf.contrib.layers.batch_norm(conv, is_training=is_training, scope='bn',
                                              variables_collections=['bn_collections'])
            if use_prelu:
                r = self.prelu(bn, 'r')
            else:
                r = tf.nn.relu(bn)
            return r
            
    def deconv3d(self, input_, output_shape, f_size, use_prelu, is_training, scope='deconv3d'):
        with tf.variable_scope(scope):
            output_dim = output_shape[-1]
            w = tf.get_variable('w', [f_size, f_size, f_size, output_dim, input_.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            deconv = tf.nn.conv3d_transpose(input_, w, output_shape, strides=[1, f_size, f_size, f_size, 1], padding='SAME')
            bn = tf.contrib.layers.batch_norm(deconv, is_training=is_training, scope='bn',
                                              variables_collections=['bn_collections'])
            if use_prelu:
                r = self.prelu(bn, 'r')
            else:
                r = tf.nn.relu(bn)
            
            return r
    
    def prelu(self, input_, scope='prelu'):
        with tf.variable_scope(scope):
            alphas = tf.get_variable('alpha', input_.get_shape()[-1], initializer=tf.constant_initializer(0.0))
            pos = tf.nn.relu(input_)
            neg = alphas * (input_ - abs(input_)) * 0.5
            return pos + neg
    
    def crop_and_concat(self, x1, x2):
        x1_shape = x1.get_shape().as_list()
        x2_shape = x2.get_shape().as_list()
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 4)
    
    @property
    def model_dir(self):
        if self.testing_during_training == True:
            prefix = "unet3d"
        else:
            prefix = "unet3d_final"
        if self.use_prelu:
            prefix += "_prelu"
        if self.use_conv_for_pooling:
            prefix += "_convpool"
        return "{}_features{}_im{}_layer{}_{}".format(prefix, self.features_root, self.im_size[0], self.layers, 
                                                      self.loss_type)