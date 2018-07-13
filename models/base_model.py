from __future__ import division
import os, re, glob
import numpy as np
from skimage.transform import resize, warp, AffineTransform
from skimage import measure
from transforms3d.euler import euler2mat
from transforms3d.affines import compose
import tensorflow as tf
import SimpleITK as sitk
import h5py

from evaluations import compute_metric

class BaseModel(object):
    def save(self, step, model_name='main'):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir, model_name)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)
        
    def load(self, model_name='main'):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir, model_name)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            return True, counter
        else:
            print("Failed to find a checkpoint")
            return False, 0
        
    def estimate_mean_std(self, is_finetune=False):
        means = []
        stds = []
        for i in range(400):
            n = np.random.choice(len(self.training_subjects))
            images, _ = self.read_training_inputs(self.training_images[n], self.training_labels[n], is_finetune=is_finetune)
            means.append(np.mean(images, axis=(0, 1, 2)))
            stds.append(np.std(images, axis=(0, 1, 2)))
        means = np.asarray(means)
        stds = np.asarray(stds)
        return np.mean(means, axis=0), np.mean(stds, axis=0)
        
    def read_training_inputs(self, image, label, augmentation=False, is_finetune=False):
        full_size = image.shape[:-1]
        im_size = self.im_size
        
        x_range = full_size[0] - im_size[0]
        y_range = full_size[1] - im_size[1]
        z_range = full_size[2] - im_size[2]
        
        # Get sampling prob
        x_offset = int(im_size[0] / 2)
        y_offset = int(im_size[1] / 2)
        z_offset = int(im_size[2] / 2)
        
        max_im = np.amax(image, axis=3)
        low_img_thresh = np.percentile(max_im, 1)
        im = max_im[x_offset : x_offset + x_range, y_offset : y_offset + y_range, z_offset : z_offset + z_range]
        la = label[x_offset : x_offset + x_range, y_offset : y_offset + y_range, z_offset : z_offset + z_range]
        if is_finetune:
            p = np.zeros((x_range, y_range, z_range), dtype=np.float32)
            p[la > 0] = 1
        else:
            # For background, p=1, for positive labels, p=6, the rest is 3
            p = np.ones((x_range, y_range, z_range), dtype=np.float32)
            p[im > low_img_thresh] = 3
            p[la > 0] = 6
        p = p.flatten() / np.sum(p)
        
        o = np.random.choice(x_range * y_range * z_range, p=p)
        
        #o = np.random.choice(x_range * y_range * z_range)
        
        x_start, y_start, z_start = np.unravel_index(o, (x_range, y_range, z_range))
        
        image_extracted = image[x_start : x_start + im_size[0], y_start : y_start + im_size[1],
                                z_start : z_start + im_size[2], :]
        label_extracted = label[x_start : x_start + im_size[0], y_start : y_start + im_size[1], z_start : z_start + im_size[2]]
        
        if is_finetune:
            image_extracted[label_extracted == 0, :] = 0
        
        if np.random.uniform() > 0.5:
            # Flip left and right
            image_extracted = image_extracted[..., ::-1, :]
            label_extracted = label_extracted[..., ::-1]
        
        nclass = self.nclass
        class_labels = self.class_labels
        
        if augmentation:
            # augmentation
            translation = [0, 0, 0] # No translation is necessary since the location is at random
            rotation = euler2mat(np.random.uniform(-5, 5) / 180.0 * np.pi, np.random.uniform(-5, 5) / 180.0 * np.pi,
                                 np.random.uniform(-5, 5) / 180.0 * np.pi, 'sxyz')
            scale = [np.random.uniform(0.8, 1.2), np.random.uniform(0.8, 1.2), np.random.uniform(0.8, 1.2)]
            warp_mat = compose(translation, rotation, scale)
            tform_coords = self.get_tform_coords(im_size)
            w = np.dot(warp_mat, tform_coords)
            w[0] = w[0] + im_size[0] / 2
            w[1] = w[1] + im_size[1] / 2
            w[2] = w[2] + im_size[2] / 2
            warp_coords = w[0:3].reshape(3, im_size[0], im_size[1], im_size[2])

            for p in range(image_extracted.shape[-1]):
                # For warp, first scale to 0 to 1, then scale back
                extracted_min = np.amin(image_extracted[..., p])
                extracted_max = np.amax(image_extracted[..., p])
                img = (image_extracted[..., p] - extracted_min) / (extracted_max - extracted_min)
                image_extracted[..., p] = warp(img, warp_coords)
                image_extracted[..., p] = image_extracted[..., p] * (extracted_max - extracted_min) + extracted_min
        
            final_labels = np.empty(im_size + [nclass], dtype=np.float32)
            for z in range(1, nclass):
                temp = warp((label_extracted == class_labels[z]).astype(np.float32), warp_coords)
                temp[temp < 0.5] = 0
                temp[temp >= 0.5] = 1
                final_labels[..., z] = temp
            final_labels[..., 0] = np.amax(final_labels[..., 1:], axis=3) == 0   
        else:
            final_labels = np.zeros(im_size + [nclass], dtype=np.float32)
            for z in range(nclass):
                final_labels[label_extracted == class_labels[z], z] = 1
        
        return image_extracted, final_labels
    
    def read_testing_inputs(self, image, nstride=1):
        full_size = image.shape[:-1]
        im_size = self.im_size
        
        # pad first
        x_stride = int(im_size[0] / nstride)
        y_stride = int(im_size[1] / nstride)
        z_stride = int(im_size[2] / nstride)
        
        x_step = int(np.ceil(full_size[0] / x_stride)) + 1 - nstride
        y_step = int(np.ceil(full_size[1] / y_stride)) + 1 - nstride
        z_step = int(np.ceil(full_size[2] / z_stride)) + 1 - nstride
        
        x_pad = (x_step - 1) * x_stride + im_size[0]
        y_pad = (y_step - 1) * y_stride + im_size[1]
        z_pad = (z_step - 1) * z_stride + im_size[2]
        
        info = {
            'full_size': full_size,
            'pad_size': (x_pad, y_pad, z_pad),
            'step': (x_step, y_step, z_step),
            'stride': (x_stride, y_stride, z_stride)
        }
        
        pad = ((0, x_pad - full_size[0]), (0, y_pad - full_size[1]), (0, z_pad - full_size[2]), (0, 0))
        image = np.pad(image, pad, mode='constant')
        
        images = np.empty([x_step * y_step * z_step] + im_size + [image.shape[-1]], dtype=np.float32)
        
        for ix in range(x_step):
            for iy in range(y_step):
                for iz in range(z_step):
                    o = ix * y_step * z_step + iy * z_step + iz
                    images[o] = image[ix * x_stride : ix * x_stride + im_size[0], iy * y_stride : iy * y_stride + im_size[1],
                                      iz * z_stride : iz * z_stride + im_size[2], :]
                    
        return images, info
        
    def get_tform_coords(self, im_size):
        coords0, coords1, coords2 = np.mgrid[:im_size[0], :im_size[1], :im_size[2]]
        coords = np.array([coords0 - im_size[0] / 2, coords1 - im_size[1] / 2, coords2 - im_size[2] / 2])
        return np.append(coords.reshape(3, -1), np.ones((1, np.prod(im_size))), axis=0)
    
    def clean_contour(self, in_contour, is_prob=False):
        if is_prob:
            pred = (in_contour >= 0.5).astype(np.float32)
        else:
            pred = in_contour
        labels = measure.label(pred)
        area = []
        for l in range(1, np.amax(labels) + 1):
            area.append(np.sum(labels == l))
        out_contour = in_contour
        out_contour[np.logical_and(labels > 0, labels != np.argmax(area) + 1)] = 0
        return out_contour

    def restore_label(self, probs, info):
        # Returns the output probability
        full_size = info['full_size']
        pad_size = info['pad_size']
        x_step, y_step, z_step = info['step']
        x_stride, y_stride, z_stride = info['stride']
        im_size = probs.shape[1 : -1]
        
        label_prob = np.zeros(pad_size + (probs.shape[-1],), dtype=np.float32)
        label_count = np.zeros(pad_size, dtype=np.float32)
        for ix in range(x_step):
            for iy in range(y_step):
                for iz in range(z_step):
                    o = ix * y_step * z_step + iy * z_step + iz
                    label_prob[ix * x_stride : ix * x_stride + im_size[0], iy * y_stride : iy * y_stride + im_size[1],
                               iz * z_stride : iz * z_stride + im_size[2], :] += probs[o]
                    label_count[ix * x_stride : ix * x_stride + im_size[0], iy * y_stride : iy * y_stride + im_size[1],
                                iz * z_stride : iz * z_stride + im_size[2]] += 1
        
        label_prob = label_prob / np.tile(np.expand_dims(label_count, axis=3), (1, 1, 1, label_prob.shape[-1]))
        label_prob = label_prob[:full_size[0], :full_size[1], :full_size[2], :]
        return label_prob
    
    def read_images(self, path, affix):
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(glob.glob(os.path.join(path, '*_t1' + affix))[0]))
        t1ce = sitk.GetArrayFromImage(sitk.ReadImage(glob.glob(os.path.join(path, '*_t1ce' + affix))[0]))
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(glob.glob(os.path.join(path, '*_t2' + affix))[0]))
        flair = sitk.GetArrayFromImage(sitk.ReadImage(glob.glob(os.path.join(path, '*_flair' + affix))[0]))
        # scale to 0 to 1
        t1 = (t1 - np.amin(t1)) / (np.amax(t1) - np.amin(t1))
        t1ce = (t1ce - np.amin(t1ce)) / (np.amax(t1ce) - np.amin(t1ce))
        t2 = (t2 - np.amin(t2)) / (np.amax(t2) - np.amin(t2))
        flair = (flair - np.amin(flair)) / (np.amax(flair) - np.amin(flair))
        # scale to 0 mean, 1 std
        #t1 = (t1 - np.mean(t1)) / np.std(t1)
        #t1ce = (t1ce - np.mean(t1ce)) / np.std(t1ce)
        #t2 = (t2 - np.mean(t2)) / np.std(t2)
        #flair = (flair - np.mean(flair)) / np.std(flair)
        images = np.stack((t1, t1ce, t2, flair), axis=-1).astype(np.float32)
        return images
    
    def read_label(self, path):
        label = sitk.GetArrayFromImage(sitk.ReadImage(glob.glob(os.path.join(path, '*_seg.nii.gz'))[0]))
        label = label.astype(np.uint8)
        return label
    
    def get_dice(self, inp1, inp2):
        eps = 1e-5
        return (np.sum(inp1 * inp2) + eps) / (np.sum(inp1) + np.sum(inp2) + 2.0 * eps) * 2.0
    
    def crop_and_mask(self, input_images, input_mask):
        # Crop and mask based on the input_mask
        nz = np.nonzero(input_mask)
        rng = (np.amin(nz[0]), np.amax(nz[0]) + 1, np.amin(nz[1]), np.amax(nz[1]) + 1, np.amin(nz[2]), np.amax(nz[2]) + 1)
        output_images = input_images[rng[0] : rng[1], rng[2] : rng[3], rng[4] : rng[5], :]
        output_mask = input_mask[rng[0] : rng[1], rng[2] : rng[3], rng[4] : rng[5]]
        output_images[output_mask == 0, :] = 0
        return output_images, rng
    
    def load_train_images(self):
        # Load all training data
        # Note that it requires about 40G memory to load all 285 subjects
        self.training_images = []
        self.training_labels = []
        for subject in self.training_subjects:
            self.training_images.append(self.read_images(subject, self.affix))
            self.training_labels.append(self.read_label(subject))
            
        if self.testing_during_training:
            self.testing_images = []
            self.testing_labels = []
            for subject in self.testing_subjects:
                self.testing_images.append(self.read_images(subject, self.affix))
                self.testing_labels.append(self.read_label(subject))
                
    def amend_model(self):
        # Add fine tune mean and std
        # Previously trained models do not have these variables so that when directly loaded, tensorflow will give an error
        with tf.variable_scope('constants'):
            self.finetune_mean = tf.get_variable('finetune_mean', [self.input_features])
            self.finetune_std = tf.get_variable('finetune_std', [self.input_features])
        
        # Update saver as well
        self.saver = tf.train.Saver(list(set(tf.trainable_variables() + tf.get_collection_ref('bn_collections'))))

    def train(self):
        self.load_train_images()
        self.sess.run(tf.global_variables_initializer())
        
        train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir, 'train'), self.sess.graph)
        if self.testing_during_training:
            test_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir, 'test'))
        
        merged = tf.summary.merge([self.loss_summary, self.accuracy_summary, self.dice_summary])
        
        counter = 0
        
        mean, std = self.estimate_mean_std()
        self.sess.run([self.mean.assign(mean), self.std.assign(std)])
        
        for epo in range(self.epoch):
            training_subjects = np.random.permutation(len(self.training_subjects))
            for f in range(len(training_subjects) // self.batch_size):
                images = np.empty((self.batch_size, self.im_size[0], self.im_size[1], self.im_size[2], self.input_features),
                                  dtype=np.float32)
                labels = np.empty((self.batch_size, self.im_size[0], self.im_size[1], self.im_size[2], self.nclass),
                                  dtype=np.float32)
                for b in range(self.batch_size):
                    order = f * self.batch_size + b
                    images[b], labels[b] = self.read_training_inputs(self.training_images[training_subjects[order]],
                                                                     self.training_labels[training_subjects[order]])
                    
                images = (images - mean) / std
                _, train_loss, summary = self.sess.run([self.optimizer, self.loss, merged],
                                                       feed_dict={ self.images: images,
                                                                   self.labels: labels,
                                                                   self.is_training: True,
                                                                   self.keep_prob: self.dropout })
                train_writer.add_summary(summary, counter)
                counter += 1
                if np.mod(counter, 1000) == 0:
                    self.save(counter)
            
            if self.testing_during_training and (np.mod(epo + 1, 20) == 0 or epo == self.epoch - 1):
                test_dice_mean = 0
                for testing_subject in range(len(self.testing_subjects)):
                    # For faster computation, do not do test augmentation with strides
                    output_label = np.argmax(self.run_test(self.testing_images[testing_subject], 1), axis=3)
                    gt_label = self.testing_labels[testing_subject]
                    for roi in range(self.nclass - 1):
                        gt = (gt_label == self.class_labels[roi + 1]).astype(np.float32)
                        model = (output_label == roi + 1).astype(np.float32)
                        test_dice_mean += self.get_dice(gt, model)
                
                test_dice_mean = test_dice_mean / len(self.testing_subjects) / (self.nclass - 1)
                test_dice_summary = tf.Summary(value=[tf.Summary.Value(tag='dice', simple_value=test_dice_mean)])
                test_writer.add_summary(test_dice_summary, counter)
                    
        # Save in the end
        self.save(counter)
        
    def finetune(self, load_from_ckpt=False):
        if load_from_ckpt:
            self.load_train_images()
            self.sess.run(tf.global_variables_initializer())
            if not self.load()[0]:
                raise Exception("No model is found, please train first")
        
        self.amend_model()
        
        finetune_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir, 'finetune_train'), self.sess.graph)
        if self.testing_during_training:
            test_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir, 'finetune_test'))
        
        merged = tf.summary.merge([self.loss_summary, self.accuracy_summary, self.dice_summary])
        
        counter = 0
        
        mean, std = self.estimate_mean_std(is_finetune=True)
        self.sess.run([self.finetune_mean.assign(mean), self.finetune_std.assign(std)])
        
        # Fine tune with 1/4 of the epoch
        for epo in range(self.epoch // 4):
            training_subjects = np.random.permutation(len(self.training_subjects))
            for f in range(len(training_subjects) // self.batch_size):
                images = np.empty((self.batch_size, self.im_size[0], self.im_size[1], self.im_size[2], self.input_features),
                                  dtype=np.float32)
                labels = np.empty((self.batch_size, self.im_size[0], self.im_size[1], self.im_size[2], self.nclass),
                                  dtype=np.float32)
                for b in range(self.batch_size):
                    order = f * self.batch_size + b
                    images[b], labels[b] = self.read_training_inputs(self.training_images[training_subjects[order]],
                                                                     self.training_labels[training_subjects[order]],
                                                                     is_finetune=True)
                    
                images = (images - mean) / std
                _, train_loss, summary = self.sess.run([self.optimizer, self.loss, merged],
                                                       feed_dict={ self.images: images,
                                                                   self.labels: labels,
                                                                   self.is_training: True,
                                                                   self.keep_prob: self.dropout })
                finetune_writer.add_summary(summary, counter)
                counter += 1
                if np.mod(counter, 1000) == 0:
                    self.save(counter, model_name='finetune')
            
            if self.testing_during_training and (np.mod(epo + 1, 20) == 0 or epo == self.epoch - 1):
                test_dice_mean = 0
                for testing_subject in range(len(self.testing_subjects)):
                    # For faster computation, do not do test augmentation with strides
                    # To test during training in this step, have to use the ground truth as the mask so that it cheats a bit
                    gt_label = self.testing_labels[testing_subject]
                    testing_images, rng = self.crop_and_mask(self.testing_images[testing_subject], gt_label)
                    crop_label = np.argmax(self.run_test(testing_images, 1, is_finetune=True), axis=3)
                    output_label = np.zeros(gt_label.shape, dtype=np.uint8)
                    output_label[rng[0] : rng[1], rng[2] : rng[3], rng[4] : rng[5]] = crop_label
                    for roi in range(self.nclass - 1):
                        gt = (gt_label == self.class_labels[roi + 1]).astype(np.float32)
                        model = (output_label == roi + 1).astype(np.float32)
                        test_dice_mean += self.get_dice(gt, model)
                
                test_dice_mean = test_dice_mean / len(self.testing_subjects) / (self.nclass - 1)
                test_dice_summary = tf.Summary(value=[tf.Summary.Value(tag='dice', simple_value=test_dice_mean)])
                test_writer.add_summary(test_dice_summary, counter)
                    
        # Save in the end
        self.save(counter, model_name='finetune')
    
    def run_test(self, testing_image, test_nstride, flip_augmentation=False, is_finetune=False):
        if is_finetune:
            mean, std = self.sess.run([self.finetune_mean, self.finetune_std])
        else:
            mean, std = self.sess.run([self.mean, self.std])
            
        test_batch = 1
        images = np.empty((test_batch, self.im_size[0], self.im_size[1], self.im_size[2], self.input_features), 
                          dtype=np.float32)
        all_images, info = self.read_testing_inputs(testing_image, test_nstride)
        
        patch_size = all_images.shape[0]
        
        pad = int(np.ceil(all_images.shape[0] / test_batch)) * test_batch
        if pad > all_images.shape[0]:
            all_images = np.pad(all_images, ((0, pad - all_images.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')
            
        all_images = (all_images - mean) / std
        
        all_probs = np.empty((all_images.shape[:-1] + (self.nclass,)), dtype=np.float32)
            
        for n in range(all_images.shape[0] // test_batch):
            for b in range(test_batch):
                images[b] = all_images[n * test_batch + b]
            
            probs = self.sess.run(self.probs, feed_dict = { self.images: images, self.is_training: True, self.keep_prob: 1 })
            if flip_augmentation:
                images = images[..., ::-1, :]
                probs_flip = self.sess.run(self.probs, feed_dict = { self.images: images, self.is_training: True,
                                                                     self.keep_prob: 1 })
                probs = (probs + probs_flip[..., ::-1, :]) / 2.0
            all_probs[n * test_batch : (n + 1) * test_batch] = probs
                
        output_label_prob = self.restore_label(all_probs[:patch_size, ...], info)
        return output_label_prob
            
    def test(self, output_path, test_with_gt, with_finetune=False):
        if not self.load()[0]:
            raise Exception("No model is found, please train first")
        
        self.testing_images = []
        self.testing_labels = []
        for subject in self.testing_subjects:
            self.testing_images.append(self.read_images(subject, self.affix))
            if test_with_gt:
                self.testing_labels.append(self.read_label(subject))
        
        if test_with_gt:
            mean_dice = []
            for roi in range(self.nclass - 1):
                mean_dice.append(0)
            # WT dice
            mean_dice.append(0)
        else:
            output_path = os.path.join(output_path, self.model_dir)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            prob_path = os.path.join(output_path, 'probs')
            if not os.path.exists(prob_path):
                os.makedirs(prob_path)
                
        if with_finetune:
            all_output_labels = []
            # Step 1
            for testing_subject in range(len(self.testing_subjects)):
                # Flip augmentation helps
                output_label_prob = self.run_test(self.testing_images[testing_subject], self.deploy_nstride, 
                                                  flip_augmentation=True)
                output_label = np.argmax(output_label_prob, axis=3)
                all_output_labels.append(output_label)
            
            # Amend model and load new weights
            self.amend_model()
            if not self.load(model_name='finetune')[0]:
                raise Exception("No model is found, please train first")
        
        for testing_subject in range(len(self.testing_subjects)):
            # Flip augmentation helps
            if with_finetune:
                testing_images, rng = self.crop_and_mask(self.testing_images[testing_subject], 
                                                         all_output_labels[testing_subject])
                crop_label_prob = self.run_test(testing_images, self.deploy_nstride, flip_augmentation=True, is_finetune=True)
                output_shape = self.testing_images[testing_subject].shape + (nclass,)
                
                output_label_prob = np.zeros(output_shape, dtype=np.float32)
                output_label_prob[..., 0] = 1 # make all background
                # Fill the cropped regions
                output_label_prob[rng[0] : rng[1], rng[2] : rng[3], rng[4] : rng[5], :] = crop_label_prob
            else:
                output_label_prob = self.run_test(self.testing_images[testing_subject], self.deploy_nstride, 
                                                  flip_augmentation=True)
            output_label = np.argmax(output_label_prob, axis=3)
            
            if test_with_gt:
                gt_label = self.testing_labels[testing_subject]
                gt_wt = gt_label > 0
                model_wt = output_label > 0
                for roi in range(self.nclass - 1):
                    gt = (gt_label == self.class_labels[roi + 1]).astype(np.float32)
                    model = (output_label == roi + 1).astype(np.float32)
                    dice = self.get_dice(gt, model)
                    mean_dice[roi] = mean_dice[roi] + dice
                mean_dice[-1] = mean_dice[-1] + self.get_dice(gt_wt, model_wt)
            else:
                subject_name = os.path.basename(self.testing_subjects[testing_subject])
                label_writing = np.empty(output_label.shape, dtype=np.uint8)
                for roi in range(self.nclass):
                    label_writing[output_label == roi] = self.class_labels[roi]
                path = self.testing_subjects[testing_subject]
                t1_img = sitk.ReadImage(glob.glob(os.path.join(path, '*_t1.nii.gz'))[0])
                label_img = sitk.GetImageFromArray(label_writing)
                label_img.CopyInformation(t1_img)
                sitk.WriteImage(label_img, os.path.join(output_path, subject_name + '.nii.gz'))
                # Save raw probs
                f_h5 = h5py.File(os.path.join(prob_path, subject_name + '.hdf5'), 'w')
                f_h5['probs'] = output_label_prob
                f_h5.close()
                
        if test_with_gt:
            for p in range(len(mean_dice)):
                mean_dice[p] = mean_dice[p] / len(self.testing_subjects)
            print(mean_dice)