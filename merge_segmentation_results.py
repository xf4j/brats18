import os, sys, glob
import numpy as np
import SimpleITK as sitk
import h5py

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception("Need at least the input data and raw data directory")
    input_path = sys.argv[1]
    raw_path = sys.argv[2]
    
    all_models = [os.path.join(input_path, f) for f in os.listdir(input_path)
                  if os.path.isdir(os.path.join(input_path, f)) and f != 'ensemble']
    
    output_path = os.path.join(input_path, 'ensemble')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    all_subjects = [os.path.basename(f).replace('.hdf5', '') for f in os.listdir(os.path.join(all_models[0], 'probs'))]
    
    for sub in all_subjects:
        for t, model in enumerate(all_models):
            f_h5 = h5py.File(os.path.join(model, 'probs', sub + '.hdf5'), 'r')
            probs = np.asarray(f_h5['probs'], dtype=np.float32)
            f_h5.close()
            
            if t == 0:
                final_probs = probs
            else:
                final_probs = final_probs + probs

        output_label = np.argmax(final_probs, axis=3)
        label_writing = np.empty(output_label.shape, dtype=np.uint8)
        
        nclass = 4
        class_labels = [0, 1, 2, 4]
        for roi in range(nclass):
            label_writing[output_label == roi] = class_labels[roi]
        
        t1_img = sitk.ReadImage(glob.glob(os.path.join(raw_path, sub, '*_t1.nii.gz'))[0])
        label_img = sitk.GetImageFromArray(label_writing)
        label_img.CopyInformation(t1_img)
        sitk.WriteImage(label_img, os.path.join(output_path, sub + '.nii.gz'))