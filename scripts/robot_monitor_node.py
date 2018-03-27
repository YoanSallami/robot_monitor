#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import sys
import argparse
import rospy
import time
import tf2_ros
from lxml import etree
from collections import deque
import numpy
import yaml
import underworlds
from std_msgs.msg import String
from underworlds.types import Entity, Mesh, Camera, MESH, Situation
from underworlds.helpers import transformations
from underworlds.tools.loader import ModelLoader
from underworlds.tools.primitives_3d import Box

EPSILON = 0.02
TF_CACHE_TIME = 5.0
DEFAULT_CLIP_PLANE_NEAR = 0.001
DEFAULT_CLIP_PLANE_FAR = 1000.0
DEFAULT_HORIZONTAL_FOV = 50.0
DEFAULT_ASPECT = 1.33333


# just for convenience
def strip_leading_slash(s):
    return s[1:] if s.startswith("/") else s


# just for convenience
def transformation_matrix(t, q):
    translation_mat = transformations.translation_matrix(t)
    rotation_mat = transformations.quaternion_matrix(q)
    return numpy.dot(translation_mat, rotation_mat)


class RobotMonitor(object):
    """
    """
    def __init__(self, ctx, source_world, target_world, urdf_file_path, model_dir_path, robot_name, perspective_frame,
                 cam_rot, reference_frame):
        """
        The constructor method
        @param ctx: The underworlds context
        @param source_world: The name of the source world
        @param source_world: The name of the target world
        @param urdf_path: The absolute path of the robot URDF model
        @param model_dir_path: The absolute path of the meshes directory
        @param reference_frame: The reference frame of the system
        """
        self.ctx = ctx
        self.source = ctx.worlds[source_world]
        self.source_world_name = source_world
        self.target = ctx.worlds[target_world]
        self.target_world_name = target_world

        self.tfBuffer = tf2_ros.Buffer(rospy.Duration(TF_CACHE_TIME), debug=False)
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.node_mapping = {self.source.scene.rootnode.id: self.target.scene.rootnode.id}

        self.already_created_node_ids = {}
        self.time_table = {}

        self.urdf_file_path = urdf_file_path
        self.model_dir_path = model_dir_path

        self.situation_map = {}

        self.robot_name = robot_name
        rospy.set_param('robot_name', robot_name)

        self.perspective_frame = perspective_frame
        self.reference_frame = reference_frame
        self.cam_rot = cam_rot
        # The map of the parent frames ordered by frame name
        self.parent_frames_map = {}
        self.model_map = {}

        self.relations_map = {}
        self.ros_pub = {"situation_log": rospy.Publisher("robot_monitor/log", String, queue_size=5)}
        self.previous_nodes_to_update = []

        self.aabb_map = {}
        self.frames_transform = {}

        self.parent_frames_map[reference_frame] = "root"
        self.parent_frames_map["base_footprint"] = reference_frame

        self.load_urdf()

    def load_urdf(self):
        """
        This function read the URDF file given in constructor and save the robot structure
        @return : None
        """
        urdf_tree = etree.parse(self.urdf_file_path)

        urdf_root = urdf_tree.getroot()

        for link in urdf_root.iter("link"):
            if link.find("visual") is not None:
                if link.find("visual").find("geometry").find("mesh") is not None:
                    path = link.find("visual").find("geometry").find("mesh").get("filename").split("/")
                    if link.find("visual").find("geometry").find("mesh").get("scale"):
                        scale_str = link.find("visual").find("geometry").find("mesh").get("scale").split(" ")
                        scale = float(scale_str[0]) * float(scale_str[1]) * float(scale_str[2])
                    else:
                        scale = 0.1
                    count = 0
                    path_str = ""
                    element = path[len(path)-1]
                    while count < len(path):
                        if element == "meshes":
                            break
                        else:
                            path_str = "/" + element + path_str
                            count += 1
                            element = path[len(path)-1-count]

                    filename = self.model_dir_path + path_str
                    try:
                        nodes_loaded = ModelLoader().load(filename, self.ctx, world=self.target_world_name, root=None,
                                                          only_meshes=True, scale=scale)
                        for n in nodes_loaded:
                            if n.type == MESH:
                                self.model_map[link.get("name")] = n.properties["mesh_ids"]
                                self.aabb_map[link.get("name")] = n.properties["aabb"]
                    except Exception as e:
                        pass
                else:
                    if link.find("visual").find("geometry").find("box") is not None:
                        mesh_ids = []
                        sizes = link.find("visual").find("geometry").find("box").get("size").split(" ")
                        box = Box.create(float(sizes[0]), float(sizes[1]), float(sizes[2]))
                        self.ctx.push_mesh(box)
                        mesh_ids.append([box.id])
                        self.model_map[link.get("name")] = mesh_ids

    def start_moving_situation(self, subject_name):
        description = "moving("+subject_name+")"
        sit = Situation(desc=description)
        self.relations_map[description] = sit.id
        self.ros_pub["situation_log"].publish("START "+description)
        try:
            self.target.timeline.update(sit)
        except Exception as e:
            rospy.logwarn("[robot_monitor] Exception occurred : " + str(e))
        return sit.id

    def end_moving_situation(self, subject_name):
        description = "moving("+subject_name+")"
        sit_id = self.relations_map[description]
        self.ros_pub["situation_log"].publish("END "+description)
        try:
            self.target.timeline.end(self.target.timeline[sit_id])
        except Exception as e:
            rospy.logwarn("[robot_monitor] Exception occurred : "+str(e))

    def filter(self):
        nodes_to_update = []
        for node in self.source.scene.nodes:
            if node != self.source.scene.rootnode:
                new_node = node.copy()
                if node.id in self.node_mapping:
                    new_node.id = self.node_mapping[node.id]
                    if new_node in self.target.scene.nodes:
                        if not numpy.allclose(self.target.scene.nodes[new_node.id].transformation, node.transformation,
                                              rtol=0, atol=EPSILON):
                            nodes_to_update.append(node)
                else:
                    self.node_mapping[node.id] = new_node.id
                    self.frames_transform[new_node.name] = new_node.transformation
                    nodes_to_update.append(new_node)

        if nodes_to_update:
            for node in nodes_to_update:
                if node.parent == self.source.scene.rootnode.id:
                    self.target.scene.nodes.update(node)
                node.parent = self.node_mapping[node.parent] if node.parent in self.node_mapping \
                    else self.target.scene.rootnode.id
            self.target.scene.nodes.update(nodes_to_update)


    def monitor_robot(self):
        """
        This method read the frames of the robot if they exist in /tf and then update the poses/3D models of
        the robot in the output world
        @return : None
        """
        try:
            nodes_to_update = []

            node = Camera(name=self.robot_name)
            node.properties["clipplanenear"] = DEFAULT_CLIP_PLANE_NEAR
            node.properties["clipplanefar"] = DEFAULT_CLIP_PLANE_FAR
            node.properties["horizontalfov"] = math.radians(DEFAULT_HORIZONTAL_FOV)
            node.properties["aspect"] = DEFAULT_ASPECT

            msg = self.tfBuffer.lookup_transform(self.reference_frame, self.perspective_frame, rospy.Time(0))
            trans = [msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]
            rot = [msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w]

            transform = transformation_matrix(trans, rot)
            node.transformation = numpy.dot(transform, self.cam_rot)

            if node.name in self.already_created_node_ids:
                node.id = self.already_created_node_ids[node.name]
                if not numpy.allclose(self.frames_transform[node.name], node.transformation, rtol=0, atol=EPSILON):
                    self.frames_transform[node.name] = node.transformation
                    nodes_to_update.append(node)

            else:
                self.already_created_node_ids[node.name] = node.id
                self.frames_transform[node.name] = node.transformation
                nodes_to_update.append(node)

            for frame in self.model_map:
                node = Mesh(name=frame)
                node.properties["mesh_ids"] = [mesh_id for mesh_id in self.model_map[frame]]
                node.properties["aabb"] = self.aabb_map[frame]

                msg = self.tfBuffer.lookup_transform(self.perspective_frame, frame, rospy.Time(0))
                trans = [msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]
                rot = [msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z,
                       msg.transform.rotation.w]

                node.transformation = transformation_matrix(trans, rot)

                node.parent = self.already_created_node_ids[self.robot_name]
                if node.name in self.already_created_node_ids:
                    node.id = self.already_created_node_ids[frame]
                    if not numpy.allclose(self.frames_transform[node.name], node.transformation, rtol=0, atol=EPSILON):
                        self.frames_transform[node.name] = node.transformation
                        nodes_to_update.append(node)
                else:
                    self.already_created_node_ids[node.name] = node.id
                    self.frames_transform[node.name] = node.transformation
                    nodes_to_update.append(node)

            for node in self.source.scene.nodes:
                if node != self.source.scene.rootnode:
                    new_node = node.copy()
                    if node.id in self.node_mapping:
                        new_node.id = self.node_mapping[node.id]
                        if new_node in self.target.scene.nodes:
                            if not numpy.allclose(self.target.scene.nodes[new_node.id].transformation, node.transformation,
                                              rtol=0, atol=EPSILON):
                                nodes_to_update.append(node)
                    else:
                        self.node_mapping[node.id] = new_node.id
                        self.frames_transform[new_node.name] = new_node.transformation
                        nodes_to_update.append(new_node)

            if not self.previous_nodes_to_update:
                if nodes_to_update:
                    self.start_moving_situation(self.robot_name)
            else:
                if not nodes_to_update:
                    self.end_moving_situation(self.robot_name)

            if nodes_to_update:
                self.target.scene.nodes.update(nodes_to_update)
            self.previous_nodes_to_update = nodes_to_update

        except (tf2_ros.TransformException, tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass

    def run(self):
        while not rospy.is_shutdown():
            self.filter()
            self.monitor_robot()

if __name__ == "__main__":

    sys.argv = [arg for arg in sys.argv if "__name" not in arg and "__log" not in arg]
    sys.argc = len(sys.argv)

    parser = argparse.ArgumentParser(description="Add in the given output world, the nodes from input "
                                                 "world and the robot agent from ROS")
    parser.add_argument("input_world", help="Underworlds input world")
    parser.add_argument("output_world", help="Underworlds output world")
    parser.add_argument("urdf_file_path", help="The path of the urdf file")
    parser.add_argument("model_dir_path", help="The path of the robot mesh directory")
    parser.add_argument("robot_name", help="The robot name")
    parser.add_argument("perspective_frame", help="The name of the robot head gaze frame")

    parser.add_argument("--cam_rot", default="0.0_0.0_0.0",
                        help="The camera rotation offset :\"<rx>_<ry>_<rz>\" in [°] ")
    parser.add_argument("--reference", default="map", help="The reference frame")
    args = parser.parse_args()

    rospy.init_node("robot_filter", anonymous=False)

    with underworlds.Context("Robot filter") as ctx:

        rx, rz, ry = [math.radians(float(i)) for i in args.cam_rot.split("_")]
        rot = transformations.euler_matrix(rx, rz, ry, 'rxyz')
        RobotMonitor(ctx, args.input_world, args.output_world, args.urdf_file_path,  args.model_dir_path,
                    args.robot_name, args.perspective_frame, rot, args.reference).run()