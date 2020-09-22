from Evaluation import Inference
from Network import *

infer = Inference()
infer.net = multi_scale_net()
infer.forward=infer.net.MSN     #选择需要的网络
infer.estimated_weight_min=1.6
infer.evaluate_random_fusion_mean_weight()
