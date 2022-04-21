
from importlib.resources import path
import json
from tkinter import N

from sfa.data_process import transformation
from sfa.data_process.transformation import lidar_to_camera_box
from sfa.data_process.kitti_data_utils import Calibration
from sfa.data_process.kitti_data_utils import  get_filtered_lidar
from sfa.data_process.kitti_bev_utils import makeBEVMap,drawRotatedBox
from sfa.data_process.kitti_dataset import KittiDataset
import sfa.config.kitti_config as cnf
from sfa.config.train_config import parse_train_configs as tcnf
from sfa.utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from sfa.test import parse_test_configs
from sfa.models.model_utils import create_model
from sfa.utils.torch_utils import _sigmoid
from sfa.utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values


from tornado import web, ioloop
import numpy as np
import cv2
import torch


class IndexHandler(web.RequestHandler):
    def get(self, *args, **kwargs):
        self.render('templates/3d_idst.html')
        # self.render('templates/upload.html')


    def post(self, *args, **kwargs):
        #获取请求参数
        [{'body': '\0\x00IEND\xaeB`\x82', 'content_type': u'image/png', 'filename': u'1.png'}]
        # try:
        img1 = self.request.files['img']

        #遍历img1
        for img in img1:
            body = img.get('body','')
            # print(body)
            content_type = img.get('content_type','')
            filename = img.get('filename','')
            leix = filename[-3:]

        #将数据存放至files目录中
        import os
        dir = os.path.join(os.getcwd(),'templates', leix, 'test.'+leix)
        dir_gx = os.path.join(os.getcwd(),'templates', 'gx', 'test.'+leix)
        dir_bq = os.path.join(os.getcwd(),'templates', 'bq', 'test.'+leix)

        #后缀txt可能是标签也可能是关系数据
        if leix == 'txt':
            #关系数据
            if str(body)[:4] == "b'P0":
                with open(dir_gx,'wb') as fw:
                    fw.write(body)
                self.write(json.dumps({'sf':'gx'}))
            else:
                with open(dir_bq,'wb') as fw:
                    fw.write(body)
                self.write(json.dumps({'sf':'bq'}))
        else:
            with open(dir,'wb') as fw:
                fw.write(body)

        if leix == 'bin':
            lidarData = np.fromfile(dir, dtype=np.float32).reshape(-1, 4)
            lidarData = get_filtered_lidar(lidarData, cnf.boundary)
            bev_map = makeBEVMap(lidarData, cnf.boundary)
            bev_map = (bev_map.transpose(1, 2, 0) * 255).astype(np.uint8)
            bev_map = cv2.resize(bev_map, (cnf.BEV_HEIGHT, cnf.BEV_WIDTH))
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
            cv2.imwrite(r'D:\tb\SFA3D-master (1)\templates\bev/bev.png', bev_map)


class kshHandler(web.RedirectHandler):
    def initialize(self, zsjg):
        self.zsjg = zsjg

    def post(self):
        bqksh()
        self.write(json.dumps({'kk':99}))

    def get(self):
       
        self.write(json.dumps({'kk': mxyc()}))


def bqksh():
    tnf = tcnf()
    ktt = KittiDataset(tnf)
    lidarData = np.fromfile(r'D:\tb\SFA3D-master (1)\templates\bin\test.bin', dtype=np.float32).reshape(-1, 4)
    calib = Calibration(r'D:\tb\SFA3D-master (1)\templates\gx\test.txt')
    labels, has_labels = ktt.get_label(r'D:\tb\SFA3D-master (1)\templates\bq\test.txt')
    if has_labels:
        labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P2)
    lidarData, labels = get_filtered_lidar(lidarData, cnf.boundary, labels)
    bev_map = makeBEVMap(lidarData, cnf.boundary)
    img_rgb = cv2.cvtColor(cv2.imread(r'D:\tb\SFA3D-master (1)\templates\png\test.png'), cv2.COLOR_BGR2RGB)

    bev_map = (bev_map.transpose(1, 2, 0) * 255).astype(np.uint8)
    bev_map = cv2.resize(bev_map, (cnf.BEV_HEIGHT, cnf.BEV_WIDTH))

    for box_idx, (cls_id, x, y, z, h, w, l, yaw) in enumerate(labels):
        # Draw rotated box
        yaw = -yaw
        y1 = int((x - cnf.boundary['minX']) / cnf.DISCRETIZATION)
        x1 = int((y - cnf.boundary['minY']) / cnf.DISCRETIZATION)
        w1 = int(w / cnf.DISCRETIZATION)
        l1 = int(l / cnf.DISCRETIZATION)

        drawRotatedBox(bev_map, x1, y1, w1, l1, yaw, cnf.colors[int(cls_id)])
    # Rotate the bev_map
    bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

    labels[:, 1:] = lidar_to_camera_box(labels[:, 1:], calib.V2C, calib.R0, calib.P2)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_rgb = show_rgb_image_with_boxes(img_rgb, labels, calib)

    out_img = merge_rgb_to_bev(img_rgb, bev_map, output_width=608)
    cv2.imwrite(r'D:\tb\SFA3D-master (1)\templates\png/out.png', out_img)
    # cv2.imshow('bev_map', out_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def mxyc():
    configs = parse_test_configs()
    model = create_model(configs)

    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()

    lidarData = np.fromfile(r'D:\tb\SFA3D-master (1)\templates\bin\test.bin', dtype=np.float32).reshape(-1, 4)
    lidarData = get_filtered_lidar(lidarData, cnf.boundary)
    bev_map = makeBEVMap(lidarData, cnf.boundary)
    bev_map = torch.from_numpy(bev_map[None])
    input_bev_map = bev_map.to(configs.device, non_blocking=True).float()

    outputs = model(input_bev_map)

    outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
    outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
    # detections size (batch_size, K, 10)
    detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                        outputs['dim'], K=configs.K)
    detections = detections.cpu().detach().numpy().astype(np.float32)
    # detections = detections.cpu().numpy().astype(np.float32)
    detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)

    detections = detections[0]  # only first batch
    # Draw prediction in the image
    bev_map = (bev_map.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
    bev_map = draw_predictions(bev_map, detections.copy(), configs.num_classes)

    # Rotate the bev_map
    bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

    img_rgb = cv2.cvtColor(cv2.imread(r'D:\tb\SFA3D-master (1)\templates\png\test.png'), cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    calib = Calibration(r'D:\tb\SFA3D-master (1)\templates\gx\test.txt')
    kitti_dets = convert_det_to_real_values(detections)
    if len(kitti_dets) > 0:
        kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
        img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)

    out_img = merge_rgb_to_bev(img_bgr, bev_map, output_width=configs.output_width)
    return cv2.imwrite(r'D:\tb\SFA3D-master (1)\templates\png/yc.png', out_img)


if __name__ == '__main__':
    settings={'debug':True}
    app = web.Application([
        (r'^/$',IndexHandler),
        (r'/ksh',kshHandler, {'zsjg':22222}),
        (r'/bev/(.*)', web.StaticFileHandler, {'path': r'D:\tb\SFA3D-master (1)\templates\bev/'}),
        # (r'^/sju_cl$',sju_clHandler),
        ('/templates/(.*)', web.StaticFileHandler, {'path': r'D:\tb\SFA3D-master (1)\templates/'}),
    ],**settings)

    #绑定监听端口号
    app.listen(8244) 

    #启动监听
    print('The server on %s is started!' % 'http://127.0.0.1:8244/')
    ioloop.IOLoop.current().start()


