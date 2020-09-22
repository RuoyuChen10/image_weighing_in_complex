from Evaluation import Inference
from Network import *

infer = Inference()
infer.net = multi_scale_net()
infer.forward=infer.net.MSN     #选择需要的网络
infer.forward=infer.net.VGG_9_pyramind_2
infer.evaluate_fusion_image()
