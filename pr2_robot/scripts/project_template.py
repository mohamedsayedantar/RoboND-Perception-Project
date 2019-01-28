#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    #print ("third")
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)
        print ("done!")


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    ##########################################
    # first part :- filtering and segmentation
    ##########################################


    # first by converting the ros message type to pcl data
    cloud_filtered = ros_to_pcl(pcl_msg)

    # due to external factors like dust in the environment we apply Outlier Removal Filter
    # taking the number of neighboring points = 4 & threshold scale factor = .00001
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(4)
    x = 0.00001
    outlier_filter.set_std_dev_mul_thresh(x)
    cloud_filtered = outlier_filter.filter()

    # using VoxelGrid Downsampling Filter to derive a point cloud that has fewer points
    # Creating a VoxelGrid filter object taking the leaf-size = .01
    vox = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # due to prior information about the location of the target in the scene
    # applying Pass Through Filter to remove useless data from the point cloud in Z $ Y axis
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.61
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.45
    axis_max = 0.45
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    # using The RANSAC algorithm to remove the table from the scene
    # by Creating the segmentation object and Setting the model
    # setting the max_distance then extracting the inliers and outliers
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance =0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    extracted_table = cloud_filtered.extract(inliers, negative=False)
    extracted_objects = cloud_filtered.extract(inliers, negative=True)

    # using a PCL library to perform a DBSCAN cluster search
    # first convert XYZRGB to XYZ then Create a cluster extraction object
    # then by Settin the tolerances for distance threshold, minimum and maximum cluster size
    # then Search the k-d tree for clusters and Extract indices for each of the discovered clusters
    white_cloud = XYZRGB_to_XYZ(extracted_objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.015)
    ec.set_MinClusterSize(1)
    ec.set_MaxClusterSize(50000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    # final step is to visualize the results in RViz!
    # by creating another point cloud of type PointCloud_PointXYZRGB
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # eventually Converting PCL data to ROS messages to Publish it
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    ros_cloud_objects = pcl_to_ros(extracted_objects)
    ros_cloud_table = pcl_to_ros(extracted_table)

    # Publish the ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)


    #########################################
    # second part :- object recognition
    #########################################


    # Classify the clusters
    detected_objects_labels = []
    detected_objects = []

    # loop to cycle through each of the segmented clusters
    for index, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster
        pcl_cluster = extracted_objects.extract(pts_list)

        # convert the cluster from pcl to ROS
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        color_hists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        normal_hists = compute_normal_histograms(normals)
        feature = np.concatenate((color_hists, normal_hists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .2
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)

    except rospy.ROSInterruptException:
        pass


# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # first we have to  Initialize the variables
    object_name = String()
    test_scene_num = Int32()
    pick_pose = Pose()
    place_pose = Pose()
    arm_name = String()
    #print ("first")

    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')
    test_scene_num.data = 2
    yaml_dict_list = []
    labels = []
    centroids = []

    # access the (x, y, z) coordinates of each point and compute the centroid
    for object in object_list:
        labels.append(object.label)
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroids.append(np.mean(points_arr, axis=0)[:3])

    # then loop through the pick list
    for i in range(0, len(object_list_param)):
        # get the object name and group from object list
        object_name.data = object_list_param[i]['name']
        object_group = object_list_param[i]['group']

        try:
            index = labels.index(object_name.data)
        except ValueError:
            print "Object not detected: %s" %object_name.data
            continue

        # get the pick pose and place pose
        pick_pose.position.x = np.asscalar(centroids[index][0])
        pick_pose.position.y = np.asscalar(centroids[index][1])
        pick_pose.position.z = np.asscalar(centroids[index][2])

        selected_dic = [element for element in dropbox_param if element['group'] == object_group][0]
        position = selected_dic.get('position')
        place_pose.position.x = position[0]
        place_pose.position.y = position[1]
        place_pose.position.z = position[2]

        # to know wich arm to be used
        selected_dic = [element for element in dropbox_param if element['group'] == object_group][0]
        arm_name.data = selected_dic.get('name')

        # Create a list of dictionaries for yaml
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        yaml_dict_list.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
#        rospy.wait_for_service('pick_place_routine')
#        print ("loop")
#        try:
#            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
#            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
#            print ("Response: ",resp.success)
#        except rospy.ServiceException, e:
#            print "Service call failed: %s"%e

    # get the output in yaml format
    yaml_filename = 'output_'+str(test_scene_num.data)+'.yaml'
    send_to_yaml(yaml_filename, yaml_dict_list)
    return

if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub   = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
