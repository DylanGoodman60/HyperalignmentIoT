# TREE CLUSTER ALIGNMENT
# Aligns clusters of images in a given folder
# Built for the data given in NVK_Nyburg provided by David
#   and pointed to by a given JSON file
# ---------------------------------------
# Dylan Goodman USRA

import numpy as np
import cv2
import json
import hypertools as hyp
import timeit
from hyp import *


start = timeit.default_timer()
times = []
CONST_CLUSTER_SIZE = 5 # The number of images to align together
CONST_FOLDER_TO_ALIGN = "Field 1/1/"

# Align function pulled from Hyperalignment developed by contextlab
# https://github.com/ContextLab/hypertools/blob/master/hypertools/tools/align.py
def align(data):
    print('start alignment')
    ##STEP 0: STANDARDIZE SIZE AND SHAPE##
    sizes_0 = [x.shape[0] for x in data]
    sizes_1 = [x.shape[1] for x in data]

    #find the smallest number of rows
    R = min(sizes_0)
    C = max(sizes_1)

    m = [np.empty((R,C), dtype=np.ndarray)] * len(data)

    for idx,x in enumerate(data):
        y = x[0:R,:]
        missing = C - y.shape[1]
        add = np.zeros((y.shape[0], missing))
        y = np.append(y, add, axis=1)
        m[idx]=y
    ##STEP 1: TEMPLATE##
    for x in range(0, len(m)):
        if x==0:
            template = np.copy(m[x])
        else:
            next = hyp.tools.procrustes(m[x], template / (x + 1))
            template += next
    template /= len(m)
    ##STEP 2: NEW COMMON TEMPLATE##
    #align each subj to the template from STEP 1
    template2 = np.zeros(template.shape)
    for x in range(0, len(m)):
        next = hyp.tools.procrustes(m[x], template)
        template2 += next
    template2 /= len(m)
    #STEP 3 (below): ALIGN TO NEW TEMPLATE
    aligned = [np.zeros(template2.shape)] * len(m)
    for x in range(0, len(m)):
        next = hyp.tools.procrustes(m[x], template2)
        aligned[x] = next
    return aligned

# align cluster of images to a max of CONST_CLUSTER_SIZE images
# Images length should not be less than CONST_CLUSTER_SIZE
def align_cluster(images):
    align_data_r = []
    align_data_g = []
    align_data_b = []
    for i in range(CONST_CLUSTER_SIZE):
        # To reduce quality and speed up alignment time, imread images with 
        # optional 2nd parameter '0' for black and white images. You can also
        # resize the images with cv2.resize(img, fx=0.5, fy=0.5) <-- halfsize
        img = cv2.imread(images[i])

        # Align RGB values seperately        
        r, g, b = cv2.split(img)
        align_data_r.append(r)
        align_data_g.append(g)
        align_data_b.append(b)

    start_align = timeit.default_timer()
    res_r = align(align_data_r)
    res_g = align(align_data_g)
    res_b = align(align_data_b)
    stop_align = timeit.default_timer()
    times.append(stop_align - start_align)
    print(str(stop_align - start_align) + " to align")

    # Combine RGB back together
    return cv2.merge((res_r[CONST_CLUSTER_SIZE-1], res_g[CONST_CLUSTER_SIZE-1], res_b[CONST_CLUSTER_SIZE-1]))

json_file = open("tree_tracking.json")
tree_data = json.load(json_file) # Loads JSON file into dictionary
tree_images = [[] for i in range(90)]
count = 0
for data in tree_data:
    filename = data['image']
    tree_id_arr = data['tree_ids']
    if CONST_FOLDER_TO_ALIGN in filename:
        for tree_id in tree_id_arr:
            tree_images[tree_id-1].append(filename)

dict_array = []
count = 0

# JSON file that points filenames to tree numbers
# filename: tree number
for img_arr in tree_images:
    count += 1
    img_arr.sort()
    if len(img_arr) < CONST_CLUSTER_SIZE: continue
    aligned_result = align_cluster(img_arr)
    out_name = 'results/fake/' + str(count) + '.jpg'

    dic = {}
    dic['image'] = out_name
    dic['tree_ids'] = count
    dict_array.append(dic)
    cv2.imwrite(out_name,aligned_result)
    cv2.waitKey(20)
    print(count)

stop = timeit.default_timer()
print("Total time: " + str(stop - start))

with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(dict_array, f, ensure_ascii=False, indent=4)