from collections import defaultdict
CLASSES = ["aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
DIM = [[690,2310], [40,50], [3,6], [5*12,25*12], [8, 2.5], [150,100], [56.5, 70], [10, 15], [36.5, 21.57], [65,100], [45, 35], [25, 30], \
      [81, 92], [23,38], [68, 18], [9,12], [43,50], [32, 91], [100,80], [18,32]]
avg_dim = defaultdict(list)
for i in range(len(CLASSES)):
    avg_dim[CLASSES[i]] = dim[i]
 FOCAL_LENGTH = 0.144
 def distance_to_camera(knownDim, calculatedDim, focal_length = FOCAL_LENGTH):
    if(knownDim[0]> knownDim[1]):
        return ((knownDim[0] * focal_length)/calculatedDim[0])
    else:
        return ((knownDim[1] * focal_length)/calculatedDim[1])
