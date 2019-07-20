import argparse
import logging
import sys
import time
import os 
from tf_pose import common
import cv2
# import math
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280,720))

path = "./images"
filenames = []
for file in os.listdir(path):
    filename = os.path.join(path,file)
    filenames.append(filename)  
filenames.sort()


body_to_dict = lambda c_fig: {'bp_{}_{}'.format(k, vec_name): vec_val 
                              for k, part_vec in c_fig.body_parts.items() 
                              for vec_name, vec_val in zip(['x', 'y', 'score'],
                                                           (part_vec.x, 1-part_vec.y, part_vec.score))}

def get_mid_pose(pose_list):
    # dict is for all people in frame
    # pose_means = []
    min_dist = 1
    min_index = None
    p_centroid = [None, None]
    # loop through each dict in the frame (each is a person)
    for i, dct in enumerate(pose_list):
        
        # loop through each post point
        # initialize a list of average x and y values for each person
        p_x_list = []
        p_y_list = []
        for k,v in dct.items():
            # print("Key: {}, Value: {}".format(k,v))
            if k.endswith("x"):
                p_x_list.append(v)
            if k.endswith("y"):
                p_y_list.append(v)
        # calculate average x and y values for mean coordinate of person
        xmean = sum(p_x_list)/len(p_x_list)
        ymean = sum(p_y_list)/len(p_y_list)

        # calc distance between them and half the frame
        dist = ((xmean-0.5)**2+(ymean - 0.5)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            min_index = i
            p_centroid = [xmean, ymean]

    # return index of pose list input where min dist is found
    pose_dict = pose_list[min_index]
    # print(p_centroid, min_index, pose_dict)
    return pose_dict, min_index, p_centroid


# arm lengh is calculated per frame
def compare_arm_angle(pose_1, cent):
    pres2, pres3, pres4 = False, False, False
    pres5, pres6, pres7 = False, False, False
    pres234arm,pres567arm = False, False
    pres8,pres11,pres1 = False, False, False

    for k,v in pose_1.items():
        if k == "bp_2_x":
            pose_2_x = v
            pres2 = True
        if k == "bp_2_y":
            pose_2_y = v
        if k == "bp_3_x":
            pose_3_x = v
            pres3 = True
        if k == "bp_3_y":
            pose_3_y = v
        if k == "bp_4_x":
            pose_4_x = v
            pres4 = True
        if k == "bp_4_y":
            pose_4_y = v
        if k == "bp_5_x":
            pose_5_x = v
            pres5 = True
        if k == "bp_5_y":
            pose_5_y = v
        if k == "bp_6_x":
            pose_6_x = v
            pres6 = True
        if k == "bp_6_y":
            pose_6_y = v
        if k == "bp_7_x":
            pose_7_x = v
            pres7 = True
        if k == "bp_7_y":
            pose_7_y = v
        if k == "bp_1_x":
            pose_1_x = v
            pres1 = True
        if k == "bp_1_y":
            pose_1_y = v
        if k == "bp_8_x":
            pose_8_x = v
            pres8 = True
        if k == "bp_8_y":
            pose_8_y = v
        if k == "bp_11_x":
            pose_11_x = v
            pres11 = True
        if k == "bp_11_y":
            pose_11_y = v

    if (pres2 and pres3 and pres4):
        length_234_arm = ((pose_4_x - cent[0])**2 + (pose_4_y - cent[1])**2)**0.5
        pres234arm = True

    if (pres5 and pres6 and pres7):
        length_567_arm = ((pose_7_x - cent[0])**2 + (pose_7_y - cent[1])**2)**0.5
        pres567arm = True

    # check both arms first before returning either of them, calculate both arm angle and return the larger range
    if (pres234arm and pres567arm and pres8 and pres1 and pres11):
        mid_back_x = (pose_8_x + pose_11_x) / 2
        mid_back_y = (pose_8_y + pose_11_y) / 2
        # back_angle = (pose_1_y - mid_back_y) / (pose_1_x - mid_back_x)
        back_angle = np.rad2deg(np.arctan2(pose_1_y - mid_back_y, pose_1_x - mid_back_x))
        arm_234_angle = np.rad2deg(np.arctan2(pose_4_y - pose_2_y, pose_4_x - pose_2_x))
        arm_567_angle = np.rad2deg(np.arctan2(pose_7_y - pose_5_y, pose_7_x - pose_5_x))

        # return the arm length and which arm it is
        if 180 >= arm_234_angle >= 0:
            return length_234_arm, "234"
        if 180 >=  arm_567_angle >= 0:
            return length_567_arm, "567"

        return (None,None)
    # if nothing is found in the psoe return empty values
    return (None,None)

    if (pres234arm and pres8 and pres1 and pres11):
        mid_back_x = (pose_8_x + pose_11_x) / 2 
        mid_back_y = (pose_8_y + pose_11_y) / 2 
        # back_angle = (pose_1_y - mid_back_y) / (pose_1_x - mid_back_x)
        back_angle = np.rad2deg(np.arctan2(pose_1_y - mid_back_y, pose_1_x - mid_back_x))
        # arm_angle = (pose_4_y - pose_2_y) / (pose_4_x - pose_2_x) 
        arm_234_angle = np.rad2deg(np.arctan2(pose_4_y - pose_2_y, pose_4_x - pose_2_x))
        if 180 >= arm_234_angle >= 0:
            return length_234_arm, "234"
        return (None,None)

    if (pres567arm and pres8 and pres1 and pres11):
        mid_back_x = (pose_8_x + pose_11_x) / 2 
        mid_back_y = (pose_8_y + pose_11_y) / 2 
        # back_angle = (pose_1_y - mid_back_y) / (pose_1_x - mid_back_x)
        back_angle = np.rad2deg(np.arctan2(pose_1_y - mid_back_y, pose_1_x - mid_back_x))
        # arm_angle = (pose_4_y - pose_2_y) / (pose_4_x - pose_2_x) 
        arm_567_angle = np.rad2deg(np.arctan2(pose_7_y - pose_5_y, pose_7_x - pose_5_x))

        if 180 >= arm_567_angle >= 0:
            return length_567_arm, "567"
        return (None,None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    # parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    # w, h = model_wh(args.resize)
    w, h = 656,368
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    max_arm_length = 0
    max_arm_frame = None
    arm_list = []
    # loop through sorted frames
    for frame_num, img in enumerate(filenames):
        frame_num += 1
        
        # estimate human poses from a single image !
        # image = common.read_imgfile("./images/{}".format(img), None, None)
        image = cv2.imread(img)
        if image is None:
            logger.error('Image can not be read, path=%s' % img)
            sys.exit(-1)

        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        # print(humans)

        humans_dict_list = []
        for hum in humans:
            c_fig = hum
            h_dict = body_to_dict(c_fig)
            humans_dict_list.append(h_dict)
        
        # print(pose_1)

        print("####################{}#########################".format(frame_num))

        ## now get pitcher from humans in image
        # get middle person / pitcher, which index of humans they are, and their centroid coordinate 
        pose_1, ind, mid_1 = get_mid_pose(humans_dict_list)
        # print(get_mid_pose)
        
        pitcher = [humans[ind]] # the drawing function requires a list in original format
        
        # get the arm length if it exists - and compare it to current arm length
        # print(pose_1,mid_1)
        length, arm = compare_arm_angle(pose_1, mid_1)
        if (length is not None):
            arm_list.append(np.asarray([frame_num,length,arm]))
            if length > max_arm_length:
                max_arm_length = length
                max_arm_frame = img
                dst = max_arm_frame.strip("./images/")

        elapsed = time.time() - t

        logger.info('inference image: %s in %.4f seconds.' % (img, elapsed))
        print("#############################################")

        # draw pose of pitcher
        image = TfPoseEstimator.draw_humans(image, pitcher, imgcopy=False)

        try:
            out.write(image)
        except Exception as e:
            logger.warning('matplitlib error, %s' % e)

        
    out.release()
    print(max_arm_frame)
    # os.rename(max_arm_frame, max_arm_frame.strip("./images/"))
    print(dst)
    print(np.vstack(arm_list))