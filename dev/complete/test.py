from launcher import Launcher
import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


IMG="/home/ftheeten/test/segmentation/test_segmentation.jpg"

obj=Launcher(IMG,p_segment=True, p_classify_keywords=True, p_timeout=120 )
obj.process()

