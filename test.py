import os
import numpy as np
import joblib
from tqdm import tqdm
import cv2
from glob import glob
import mahotas
from scipy.spatial.distance import cdist
from skimage.feature import hog
import time
# wh_ratios = (0.5, 0.8, 1.25, 2.0)
wh_ratios = (0.5, 1.0, 2.0)
slide_config = {
	# 'strides': [0.01, 0.05, 0.15], 
	# 'wid': [0.2, 0.3, 0.5],	
	'strides': [ 0.05, 0.08, 0.10], 
	'wid': [ 0.3, 0.5, 0.8],	
}
def sliding_window(image):
	for stride, w in zip(*slide_config.values()):
		objw = int(w*image.shape[1])
		for whr in wh_ratios:
			h = w/whr
			objh = int(h*image.shape[0])
			# strx, stry =int(stride*image.shape[1]), int(stride*image.shape[0])
			for y in range(0, image.shape[0]-objh, int(stride*image.shape[0])):
				for x in range(0, image.shape[1]-objw, int(stride*image.shape[1])):
					yield((x, y, objw, objh), image[y: y+objh, x: x+objw])


# https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression_fast(boxes, overlapThresh):
	if len(boxes) == 0:
		return []
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	pick = []
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		overlap = (w * h) / area[idxs[:last]]
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	return boxes[pick].astype("int")

def save_label(save_dir, name, boxes):
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	with open(os.path.join(save_dir, f'{name}.txt'), 'w') as fw:
		res = []
		for box in boxes:
			res.append(' '.join([str(i) for i in box]))
		fw.write('\n'.join(res))
DATA_NAME = \
'UTTQ'

FEATURE_TYPE = \
'HOG'
'hist'
'SIFT'
'ZERNIKE'	

orientations = 9
pixels_per_cell = (4, 4)
cells_per_block = (2, 2)
in_size = (256, 256)
def resize_gray(image):
    im = cv2.resize(image, in_size)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im

sift = cv2.SIFT_create()

def sift_des(image):
    # if image.ndim>2:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(image, None)
    return des
extract = {
    'HOG': lambda gray : hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True),
    'SIFT': lambda gray: sift_des(gray),
    'hist': lambda gray: cv2.calcHist([gray], [0], None, [256], [0,256]).squeeze(),
    'ZERNIKE': lambda gray:  mahotas.features.zernike_moments(gray, 181, 10 )
}

def create_feature_bow(image_descriptors, BoW, num_cluster):
    X_features = []
    for i in range(len(image_descriptors)):
        features = np.array([0] * num_cluster)
        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)
            argmin = np.argmin(distance, axis = 1)
            for j in argmin:
                features[j] += 1
        X_features.append(features)
    return X_features

SIFT_CONFIG = joblib.load(f'./sfit_bow/{DATA_NAME}.pkl')
for MODEL_NAME in [
        'RandomForestClassifier', 
        'DecisionTreeClassifier',
        'SVC',
        # 'KNN',
        # 'AdaBoostClassifier'
        ]:
    save_name = f'{DATA_NAME}_{in_size[0]}_{FEATURE_TYPE}_{MODEL_NAME}'
    print(save_name)
    model = joblib.load(f'./{save_name}.skm')


    # demo_images = []
    # demo_labels = []
    for image_path in tqdm(glob('./images/*.jpeg')):
        # print(image_path)
        image = cv2.imread(image_path)
        boxes = []
        start_time = time.time()
        for label, pad in sliding_window(image):
            gray = resize_gray(pad)
            fd = extract[FEATURE_TYPE](gray)
    
            if FEATURE_TYPE == 'SIFT':
                fd = create_feature_bow([fd], SIFT_CONFIG['BoW'], SIFT_CONFIG['num_cluster'])[0]
            
            pr = model.predict([fd])
            if pr[0] == 1:
                boxes.append([1, 1.0, *label])
        
        new_boxes = np.array(boxes)
        new_boxes[:, 2] += new_boxes[:, 0]
        new_boxes[:, 3] += new_boxes[:, 1]
        filtered_boxes = non_max_suppression_fast(new_boxes, 0.5)
        filtered_boxes[:, 2] -= filtered_boxes[:, 0]
        filtered_boxes[:, 3] -= filtered_boxes[:, 1]
        end_time = time.time()
        print(" time inference: ", end_time-start_time)
        name = os.path.basename(image_path).split('.')[0]
        save_label( f'./detections/{DATA_NAME}/{MODEL_NAME}', name, filtered_boxes)