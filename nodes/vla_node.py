# 1st Class Libraries
import sys
from math import pi

# OpenVLA Dependencies
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

# ROS Dependencies
import rospy
import moveit_commander
from geometry_msgs.msg import Pose


# Load Processor & VLA
model = "openvla/openvla-7b"
processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model, 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

# Init ROS Commander
moveit_commander.roscpp_initialize(sys.argv)
robot = moveit_commander.RobotCommander()

# Init ROS node
pose_goal = Pose()
rospy.init_node('vla_node')
r = rospy.Rate(1)

# Init Planning Scene
scene = moveit_commander.PlanningSceneInterface()
group_name = "widowx_arm"
group = moveit_commander.MoveGroupCommander(group_name)

while not rospy.is_shutdown():

    # Grab image input & format prompt
    image: Image.Image = Image.open('/home/sensethreat/openvla_ws/src/openvla_demo/res/frame.jpeg')
    prompt = "In: What action should the robot take to put the bottle on top of the box?\nOut:"
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    action = action.tolist()

    pose_goal.position.x = 0
    pose_goal.position.y = 0.5
    pose_goal.position.z = 0
    pose_goal.orientation.x = pi/4
    pose_goal.orientation.y = 0
    pose_goal.orientation.z = 0
    pose_goal.orientation.w = 1

    # Plan IK
    group.set_pose_target(pose_goal)
    if group.go(wait=True):
        rospy.loginfo("Executing.")
        group.execute(group.plan(), wait=True)
        group.stop()
    
    # rospy.logwarn(group.get_joints())
    rospy.logwarn(group.has_end_effector_link())
    group.clear_pose_targets()
    r.sleep()