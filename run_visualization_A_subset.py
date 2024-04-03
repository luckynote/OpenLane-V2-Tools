import os
import numpy as np
import matplotlib.pyplot as plt

from openlanev2.io import io
import cv2

from alive_progress import alive_bar
from openlanev2.dataset import Collection, Frame
from openlanev2.visualization import draw_annotation_bev, draw_annotation_pv, assign_attribute, assign_topology



root_path = './data/OpenLane-V2'

collection = Collection(root_path, 'data_dict_subset_A_train')
is_process_video_name = ""
out = None

with alive_bar(len(collection.keys)) as bar:
    for image_path_str in collection.keys:
        print(image_path_str)
        pv_bev_path = os.path.join(root_path, image_path_str[0], image_path_str[1], 'pv_bev')
        if not os.path.exists(pv_bev_path):
            os.mkdir(pv_bev_path)

        pv_bev_name = image_path_str[2]+'.jpg'

        frame = collection.get_frame_via_identifier(image_path_str)

        # for k in frame.get_camera_list():
        #     print(k)

        annotations = frame.get_annotations()

        annotations = assign_attribute(annotations)
        annotations = assign_topology(annotations)

        image_bev = draw_annotation_bev(
            annotations, 
            with_attribute=True,
        )

        img_pvs = {}
        for camera in frame.get_camera_list():
            meta = {
                'intrinsic': frame.get_intrinsic(camera),
                'extrinsic': frame.get_extrinsic(camera),
            }
            image_pv = draw_annotation_pv(
                camera, 
                frame.get_rgb_image(camera), 
                annotations,
                meta['intrinsic'],
                meta['extrinsic'],
                with_attribute=True, 
                with_topology=True,
            )

            img_pvs[camera] = image_pv

        if image_bev.dtype == np.int32:
            # 将图像深度转换为 8 位无符号整数
            image_bev = image_bev.astype(np.uint8)
        image_bev = cv2.cvtColor(image_bev, cv2.COLOR_RGB2BGR)

        scale = 0.5
        img_ring_front_left = cv2.resize(img_pvs["ring_front_left"], None, fx=scale,  fy=scale)
        img_ring_front_center_h = img_ring_front_left.shape[0]
        img_ring_front_center_w = int((img_ring_front_center_h/img_pvs["ring_front_center"].shape[0])*img_pvs["ring_front_center"].shape[1])

        img_ring_front_center = cv2.resize(img_pvs["ring_front_center"], (img_ring_front_center_w, img_ring_front_center_h))

        img_ring_front_right = cv2.resize(img_pvs["ring_front_right"], None, fx=scale,  fy=scale)

        new_img_w = img_ring_front_left.shape[1]*2+img_ring_front_center.shape[1]

        second_img_scale = new_img_w*0.25/img_pvs["ring_side_left"].shape[1]

        img_ring_side_left = cv2.resize(img_pvs["ring_side_left"], None, fx=second_img_scale,  fy=second_img_scale)
        img_ring_rear_left = cv2.resize(img_pvs["ring_rear_left"], None, fx=second_img_scale,  fy=second_img_scale)
        img_ring_rear_right = cv2.resize(img_pvs["ring_rear_right"], None, fx=second_img_scale,  fy=second_img_scale)
        img_ring_side_right = cv2.resize(img_pvs["ring_side_right"], None, fx=second_img_scale,  fy=second_img_scale)

        new_img_h = img_ring_front_left.shape[0]+img_ring_side_left.shape[0]

        scale = new_img_h/image_bev.shape[0]
        image_bev = cv2.resize(image_bev, None, fx=scale,  fy=scale)
        cv2.circle(image_bev, (image_bev.shape[1]//2, image_bev.shape[0]//2), 5, (0,0,255), -1)

        new_img = np.zeros((new_img_h, new_img_w+image_bev.shape[1], 3), dtype=np.uint8)

        new_img[:img_ring_front_left.shape[0], : img_ring_front_left.shape[1]] = img_ring_front_left
        new_img[:img_ring_front_left.shape[0], img_ring_front_left.shape[1]: img_ring_front_left.shape[1]+img_ring_front_center.shape[1]] = img_ring_front_center
        new_img[:img_ring_front_left.shape[0], img_ring_front_left.shape[1]+img_ring_front_center.shape[1]: img_ring_front_left.shape[1]*2+img_ring_front_center.shape[1]] = img_ring_front_right

        new_img[img_ring_front_left.shape[0]:, : img_ring_side_left.shape[1]] = img_ring_side_left
        new_img[img_ring_front_left.shape[0]:, img_ring_side_left.shape[1]: img_ring_side_left.shape[1]*2] = img_ring_rear_left
        new_img[img_ring_front_left.shape[0]:, img_ring_side_left.shape[1]*2: img_ring_side_left.shape[1]*3] =  img_ring_rear_right
        new_img[img_ring_front_left.shape[0]:, img_ring_side_left.shape[1]*3: img_ring_side_left.shape[1]*4] =  img_ring_side_right

        new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
        new_img[:, img_ring_front_left.shape[1]*2+img_ring_front_center.shape[1]: img_ring_front_left.shape[1]*2+img_ring_front_center.shape[1]+image_bev.shape[1]] = image_bev

        cv2.imwrite(os.path.join(pv_bev_path, pv_bev_name), new_img)

        if image_path_str[1] != is_process_video_name:
            if out is not None: out.release()

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(os.path.join(root_path, image_path_str[0], image_path_str[1], "pv_bev.avi"), fourcc, 1.0, (new_img.shape[1],new_img.shape[0]),True)
            is_process_video_name = image_path_str[1]

        out.write(new_img)

        bar()
