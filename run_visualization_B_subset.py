import os
import numpy as np
import matplotlib.pyplot as plt

from openlanev2.io import io
import cv2

from alive_progress import alive_bar
from openlanev2.dataset import Collection, Frame
from openlanev2.visualization import draw_annotation_bev, draw_annotation_pv, assign_attribute, assign_topology



root_path = './data/OpenLane-V2'

collection = Collection(root_path, 'data_dict_subset_B_train')

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

        # 创建一个新的空白图像，高度为所有图像高度之和，宽度为其中最宽图像的宽度乘以列数
        max_width = max(frame.get_rgb_image(k).shape[1] for k in frame.get_camera_list())
        max_height = max(frame.get_rgb_image(k).shape[0] for k in frame.get_camera_list())

        scale = max_height*2/image_bev.shape[0]

        if image_bev.dtype == np.int32:
            # 将图像深度转换为 8 位无符号整数
            image_bev = image_bev.astype(np.uint8)
        image_bev = cv2.cvtColor(image_bev, cv2.COLOR_RGB2BGR)
        image_bev = cv2.resize(image_bev, None, fx=scale,  fy=scale)
        cv2.circle(image_bev, (image_bev.shape[1]//2, image_bev.shape[0]//2), 5, (0,0,255), -1)
        new_img = np.zeros((max_height*2, max_width * 3+image_bev.shape[1], 3), dtype=np.uint8)

        new_img[:max_height*1, : max_width*1] = img_pvs["CAM_FRONT_LEFT"]
        new_img[:max_height*1, max_width*1: max_width*2] = img_pvs["CAM_FRONT"]
        new_img[:max_height*1, max_width*2: max_width*3] = img_pvs["CAM_FRONT_RIGHT"]
        new_img[max_height*1:max_height*2, : max_width*1] = img_pvs["CAM_BACK_LEFT"]
        new_img[max_height*1:max_height*2, max_width*1: max_width*2] = img_pvs["CAM_BACK"]
        new_img[max_height*1:max_height*2, max_width*2: max_width*3] =  img_pvs["CAM_BACK_RIGHT"]

        new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
        new_img[:image_bev.shape[0], max_width*3: max_width*3+image_bev.shape[1]] = image_bev
        new_img = cv2.resize(new_img, None, fx=0.5,  fy=0.5)

        cv2.imwrite(os.path.join(pv_bev_path, pv_bev_name), new_img)

        if image_path_str[1] != is_process_video_name:
            if out is not None: out.release()

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(os.path.join(root_path, image_path_str[0], image_path_str[1], "pv_bev.avi"), fourcc, 1.0, (new_img.shape[1],new_img.shape[0]),True)
            is_process_video_name = image_path_str[1]

        out.write(new_img)

        bar()
