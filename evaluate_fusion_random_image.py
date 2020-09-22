from Evaluation import Inference

infer = Inference()
infer.forward=infer.net.VGG_9_pyramind_2
infer.estimated_weight_min=2.1
infer.evaluate_random_fusion_mean_weight()
