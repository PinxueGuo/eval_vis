import matplotlib.pyplot as plt
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


# load ytbvis GT json
annFile = "/home/ubuntu/researches/DATA/MinVIS/BURST/annotations/val/uncommon_classes_coco.json"
# annFile = "/home/ubuntu/researches/DATA/MinVIS/UVOv1.0/VideoDenseSet/UVO_video_val_dense_with_label_1class.json"
ytvisGT = YTVOS(annFile)


# load ytbvis results json
resFile = "/home/ubuntu/researches/DATA/MinVIS/results_uncommon.json"
# resFile = "/home/ubuntu/researches/MinVIS/output/minvis_ytb19TrainedBinaryLoss_UVOtest/inference/results.json"
ytbvisDt = ytvisGT.loadRes(resFile)


# eval
cocoEval = YTVOSeval(ytvisGT,ytbvisDt)
max_dets_per_image = [1, 10, 100]  # Default from COCOEval
cocoEval.params.maxDets = max_dets_per_image

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()