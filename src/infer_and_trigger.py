# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt

import numpy

from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

# Grab image input & format prompt
image: Image.Image = Image.open('frame.jpeg')
prompt = "In: What action should the robot take to put the bottle on top of the box?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

actionList = action.tolist()
position = actionList[0:3]
orientation = actionList[3:6]
gripper = actionList[6] > 0.5

numpy.set_printoptions(precision=5)
print("Raw output: ", actionList)
print("Position     [x: %f, y: %f, z: %f]" % (position[0], position[1], position[2]))
print("Orientation  [x: %f, y: %f, z: %f]" % (orientation[0], orientation[1], orientation[2]))
print("Gripper: ", "Open" if gripper else "Closed")

# Publish to '/target_pose'
# Un-normalization required

# import rospy
# from geometry_msgs.msg import PoseStamped
# from scipy.spatial.transform import Rotation

# goal_pose = PoseStamped()
# goal_pose.pose.position.x = position[0]
# goal_pose.pose.position.y = position[1]
# goal_pose.pose.position.z = position[2]
# goal_pose.pose.orientation.x = orientation[0]
# goal_pose.pose.orientation.y = orientation[1]
# goal_pose.pose.orientation.z = orientation[2]

# pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=10)
