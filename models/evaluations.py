import numpy as np
from scipy import ndimage, spatial
import dicom
from skimage import measure

def compute_metric(gt, model, voxel_spacing):
    dice = np.sum(gt * model) / (np.sum(gt) + np.sum(model)) * 2.0
    msd, hd95 = calculate_distance(gt, model, voxel_spacing)
    return dice, msd, hd95
    #return dice, 0, 0

def get_points(mask, spacing):
    points = []
    if np.amax(mask) > 0:
        contours = measure.find_contours(mask.astype(np.float32), 0.5)
        for contour in contours:
            con = contour.ravel().reshape((-1, 2)) # con[0] is along x (last dimension of mask)
            for p in con:
                points.append([p[1] * spacing[0], p[0] * spacing[1]])
    return np.asarray(points, dtype=np.float32)
    
def calculate_distance(mask1, mask2, voxel_spacing):
    first_slice = True
    for z in range(mask1.shape[0]):
        points1 = get_points(mask1[z], voxel_spacing[1 : 3])
        points2 = get_points(mask2[z], voxel_spacing[1 : 3])
        if points1.shape[0] > 0 and points2.shape[0] > 0:
            dists = spatial.distance.cdist(points1, points2, metric='euclidean')
            if first_slice:
                dist12 = np.amin(dists, axis=1)
                dist21 = np.amin(dists, axis=0)
                first_slice = False
            else:
                dist12 = np.append(dist12, np.amin(dists, axis=1))
                dist21 = np.append(dist21, np.amin(dists, axis=0))
    # Mean surface distance
    msd = (np.mean(dist12) + np.mean(dist21)) / 2.0
    # Hausdorff distance
    hd = (np.amax(dist12) + np.amax(dist21)) / 2.0
    # 95 Hausdorff distance
    hd95 = (np.percentile(dist12, 95) + np.percentile(dist21, 95)) / 2.0
    return msd, hd95