import os, glob, sys, csv
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import scipy
import SimpleITK as sitk

def read_seg(path, is_train):
    if is_train:
        label = sitk.GetArrayFromImage(sitk.ReadImage(glob.glob(os.path.join(path, '*_seg.nii.gz'))[0]))
    else:
        label = sitk.GetArrayFromImage(sitk.ReadImage(path))
    label = label.astype(np.uint8)
    return label

def get_size_length(label):
    size = np.sum(label)
    b = np.gradient(label)
    length = np.sum(np.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]))
    return size, length

def get_features(label, age, status):
    features = []
    s, l = get_size_length((label == 1).astype(np.float32))
    features.append(s)
    features.append(l)
    s, l = get_size_length((label == 2).astype(np.float32))
    features.append(s)
    features.append(l)
    s, l = get_size_length((label == 4).astype(np.float32))
    features.append(s)
    features.append(l)
    features.append(age)
    if status == 'NA':
        features.append(0)
        features.append(0)
    elif status == 'GTR':
        features.append(1)
        features.append(0)
    elif status == 'STR':
        features.append(0)
        features.append(1)
    return features

if __name__ == '__main__':
    training_path = '../TrainingData'
    validation_path = '../ValidationData'
    validation_seg_path = './output/validation/ensemble'
    output_file = './output/validation/validation_survival.csv'
    
    # Get training subject paths and training data
    training_subjects_paths = []
    for dirpath, dirnames, files in os.walk(training_path):
        if os.path.basename(dirpath)[0:7] == 'Brats18':
            training_subjects_paths.append(dirpath)
    training_subjects = [os.path.basename(p) for p in training_subjects_paths]
    training_survival_data = {}
    with open(os.path.join(training_path, 'survival_data.csv'), 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] in training_subjects:
                training_survival_data[row[0]] = (row[1], row[2], row[3])
    
    X = []
    y = []
    for path in training_subjects_paths:
        subject = os.path.basename(path)
        if subject in training_survival_data.keys():
            features = get_features(read_seg(path, True), training_survival_data[subject][0], 
                                    training_survival_data[subject][2])
            X.append(features)
            y.append(training_survival_data[subject][1])
    
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    
    # Train a linear regression model
    linear_reg_model = linear_model.LinearRegression(normalize=True)
    linear_reg_model.fit(X, y)
    print("Training error: " + str(linear_reg_model.score(X, y)))
    
    scaler = StandardScaler()
    scaler.fit(X)
    X2 = scaler.transform(X)
    model_for_analysis = linear_model.LinearRegression(normalize=False)
    model_for_analysis.fit(X2, y)
    print(model_for_analysis.coef_)
    print(model_for_analysis.intercept_)
    for n in range(9):
        print(scipy.stats.pearsonr(X2[:, n], y))
    
    # Use this model for validation data
    validation_subjects_paths = [os.path.join(validation_seg_path, f) for f in os.listdir(validation_seg_path)]
    validation_subjects = [os.path.basename(p).replace('.nii.gz', '') for p in validation_subjects_paths]
    validation_survival_data = {}
    with open(os.path.join(validation_path, 'survival_evaluation.csv'), 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] in validation_subjects:
                validation_survival_data[row[0]] = (row[1], row[2])
    
    X_validation = []
    X_keys = []
    for path in validation_subjects_paths:
        subject = os.path.basename(path).replace('.nii.gz', '')
        if subject in validation_survival_data.keys():
            features = get_features(read_seg(path, False), validation_survival_data[subject][0],
                                    validation_survival_data[subject][1])
            
            X_keys.append(subject)
            X_validation.append(features)
    X_validation = np.asarray(X_validation, dtype=np.float32)
    y_validation = linear_reg_model.predict(X_validation)
    
    # Write the results
    with open(output_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(X_keys)):
            writer.writerow([X_keys[i], y_validation[i]])