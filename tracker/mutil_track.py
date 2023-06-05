"""
Only track a video or image seqs, without evaluate
"""
import datetime

import numpy as np
import torch
import cv2
from PIL import Image
import tqdm

import argparse
import os
from time import gmtime, strftime
from tracker.timer import Timer
import yaml
import trackeval

from tracker.basetrack import BaseTracker  # for framework
from tracker.deepsort import DeepSORT
from tracker.bytetrack import ByteTrack
from tracker.c_biou_tracker import C_BIoUTracker
from tracker.uavmot import UAVMOT
from tracker.botsort import BoTSORT


try:  # import package that outside the tracker folder  For yolo v7
    import sys

    sys.path.append(os.getcwd())
    from models.experimental import attempt_load
    from tracker.evaluate import evaluate
    from utils.torch_utils import select_device, time_synchronized, TracedModel

    print('Note: running yolo v7 detector')
except:
    pass

SAVE_FOLDER = 'multi_track_result'  # NOTE: set your save path here
CATEGORY_DICT = {
    0: 'pedestrain',
    1: 'people',
    2: 'bicycle',
    3: 'car',
    4: 'van',
    5: 'truck',
    6: 'tricycle',
    7: 'awning-tricycle',
    8: 'bus',
    9: 'motor'}  # NOTE: set the categories in your videos here,
# format: class_id(start from 0): class_name

timer = Timer()
seq_fps = []  # list to store time used for every seq


def init_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=bool, default=False, help='traced model of YOLO v7')
    parser.add_argument('--img_size', type=int, default=1280, help='[train, test] image sizes')
    """For tracker"""
    # model path
    parser.add_argument('--reid_model_path', type=str, default='./weights/ckpt.t7', help='path for reid model path')
    parser.add_argument('--dhn_path', type=str, default='./weights/DHN.pth', help='path of DHN path for DeepMOT')
    # threshs
    parser.add_argument('--conf_thresh', type=float, default=0.45, help='filter tracks')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='thresh for NMS')
    parser.add_argument('--iou_thresh', type=float, default=0.55, help='IOU thresh to filter tracks')
    # other options
    parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--gamma', type=float, default=0.1, help='param to control fusing motion and apperance dist')
    parser.add_argument('--kalman_format', type=str, default='default',
                        help='use what kind of Kalman, default, naive, strongsort or bot-sort like')
    parser.add_argument('--min_area', type=float, default=150, help='use to filter small bboxs')
    opts = parser.parse_args()
    return opts


def mutil_track(tracker1, tracker2, tracker3, obj_path, filename):
    opts = init_opts()
    TRACKER_DICT = {
        'SORT': BaseTracker,
        'DeepSORT': DeepSORT,
        'ByteTrack': ByteTrack,
        'C-BIoUTrack': C_BIoUTracker,
        'UAVMOT': UAVMOT,
        'BotSORT': BoTSORT,
    }  # dict for trackers, key: str, value: class(BaseTracker)

    """
    1. load model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load('./weights/best.pt', map_location=device)
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()  # for yolo v7

    if not os.path.exists(SAVE_FOLDER):  # demo save to a particular folder
        os.makedirs(SAVE_FOLDER)
        os.makedirs(os.path.join(SAVE_FOLDER, 'result_images'))

    """
    2. load videos or images
    """
    # if read video, then put every frame into a queue
    # if read image seqs, the same as video
    resized_images_queue = []  # List[torch.Tensor] store resized images
    images_queue = []  # List[torch.Tensor] store origin images

    # check path
    assert os.path.exists(obj_path), 'the path does not exist! '
    obj, get_next_frame = None, None  # init obj
    if 'mp4' in obj_path:  # if it is a video
        obj = cv2.VideoCapture(obj_path)
        get_next_frame = lambda _: obj.read()
    else:
        obj = my_queue(os.listdir(obj_path))
        get_next_frame = lambda _: obj.pop_front()

    """
    3. start tracking
    """
    tracker1_name = tracker1
    tracker2_name = tracker2
    tracker3_name = tracker3
    tracker1 = TRACKER_DICT[tracker1_name](opts, frame_rate=30, gamma=opts.gamma)
    tracker2 = TRACKER_DICT[tracker2_name](opts, frame_rate=30, gamma=opts.gamma)
    tracker3 = TRACKER_DICT[tracker3_name](opts, frame_rate=30, gamma=opts.gamma)

    now = datetime.datetime.now()
    folder_name1 = now.strftime("%m_%d_%H_%M")
    folder_name1 = str.format(tracker1_name) + '_' + folder_name1
    folder_name2 = now.strftime("%m_%d_%H_%M")
    folder_name2 = str.format(tracker2_name) + '_' + folder_name2
    folder_name3 = now.strftime("%m_%d_%H_%M")
    folder_name3 = str.format(tracker3_name) + '_' + folder_name3
    seqs = []
    results1 = []  # store current seq results
    results2 = []  # store current seq results
    results3 = []  # store current seq results
    point_set1 = []
    point_set2 = []
    point_set3 = []
    id_set1 = []
    id_set2 = []
    id_set3 = []
    track_len1 = []
    track_len2 = []
    track_len3 = []
    frame_id = 0
    min_area = 150
    max_trajectory_len = 18

    while True:
        print(f'----------processing frame {frame_id}----------')
        # end condition
        is_valid, img0 = get_next_frame(None)  # img0: (H, W, C)

        if not is_valid:
            break  # end of reading
        # convert BGR to RGB and to (C, H, W)
        img = resize_a_frame(img0, [1280, 1280])

        timer.tic()  # start timing this img
        img = img.unsqueeze(0)  # （C, H, W) -> (bs == 1, C, H, W)
        out = model(img.to(device))  # model forward
        out = out[0]  # NOTE: for yolo v7

        if len(out.shape) == 3:  # case (bs, num_obj, ...)
            # out = out.squeeze()
            # NOTE: assert batch size == 1
            out = out.squeeze(0)
        # remove some low conf detections
        out = out[out[:, 4] > 0.001]

        # NOTE: yolo v7 origin out format: [xc, yc, w, h, conf, cls0_conf, cls1_conf, ..., clsn_conf]
        cls_conf, cls_idx = torch.max(out[:, 5:], dim=1)
        # out[:, 4] *= cls_conf  # fuse object and cls conf
        out[:, 5] = cls_idx
        out = out[:, :6]

        current_tracks1 = tracker1.update(out, img0)  # List[class(STracks)]
        # save results
        cur_tlwh1, cur_id1, cur_cls1 = [], [], []
        for trk1 in current_tracks1:
            bbox1 = trk1.tlwh
            id1 = trk1.track_id
            cls1 = trk1.cls

            # filter low area bbox
            if bbox1[2] * bbox1[3] > min_area:
                cur_tlwh1.append(bbox1)
                cur_id1.append(id1)
                cur_cls1.append(cls1)
                # results.append((frame_id + 1, id, bbox, cls))

        results1.append((frame_id + 1, cur_id1, cur_tlwh1, cur_cls1))
        for tlwh, id in zip(cur_tlwh1, cur_id1):
            x, y, w, h = tlwh
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            point_set1.append((center_x, center_y))
            id_set1.append(id)
        track_len1.append(len(cur_id1))
        if frame_id >= max_trajectory_len:
            remove_len1 = track_len1[frame_id - max_trajectory_len]
            point_set1 = point_set1[remove_len1:]
            id_set1 = id_set1[remove_len1:]
        pointed_img1 = plot_center_point(img0, point_set1, id_set1)
        # plot_img(img0, frame_id, [cur_tlwh1, cur_id1, cur_cls1],
        #          save_dir=os.path.join(SAVE_FOLDER, 'result_images', folder_name1))
        plot_img(pointed_img1, frame_id, [cur_tlwh1, cur_id1, cur_cls1],
                 save_dir=os.path.join(SAVE_FOLDER, 'result_images', folder_name1))

        current_tracks2 = tracker2.update(out, img0)  # List[class(STracks)]
        cur_tlwh2, cur_id2, cur_cls2 = [], [], []
        for trk2 in current_tracks2:
            bbox2 = trk2.tlwh
            id2 = trk2.track_id
            cls2 = trk2.cls

            # filter low area bbox
            if bbox2[2] * bbox2[3] > min_area:
                cur_tlwh2.append(bbox2)
                cur_id2.append(id2)
                cur_cls2.append(cls2)
                # results.append((frame_id + 1, id, bbox, cls))

        results2.append((frame_id + 1, cur_id2, cur_tlwh2, cur_cls2))
        for tlwh, id in zip(cur_tlwh2, cur_id2):
            x, y, w, h = tlwh
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            point_set2.append((center_x, center_y))
            id_set2.append(id)
        track_len2.append(len(cur_id2))
        if frame_id >= max_trajectory_len:
            remove_len2 = track_len2[frame_id - max_trajectory_len]
            point_set2 = point_set2[remove_len2:]
            id_set2 = id_set2[remove_len2:]
        pointed_img2 = plot_center_point(img0, point_set2, id_set2)
        # plot_img(img0, frame_id, [cur_tlwh2, cur_id2, cur_cls2],
        #          save_dir=os.path.join(SAVE_FOLDER, 'result_images', folder_name2))
        plot_img(pointed_img2, frame_id, [cur_tlwh2, cur_id2, cur_cls2],
                 save_dir=os.path.join(SAVE_FOLDER, 'result_images', folder_name2))

        current_tracks3 = tracker3.update(out, img0)  # List[class(STracks)]
        cur_tlwh3, cur_id3, cur_cls3 = [], [], []
        for trk3 in current_tracks3:
            bbox3 = trk3.tlwh
            id3 = trk3.track_id
            cls3 = trk3.cls

            # filter low area bbox
            if bbox3[2] * bbox3[3] > min_area:
                cur_tlwh3.append(bbox3)
                cur_id3.append(id3)
                cur_cls3.append(cls3)
                # results.append((frame_id + 1, id, bbox, cls))

        results3.append((frame_id + 1, cur_id3, cur_tlwh3, cur_cls3))
        for tlwh, id in zip(cur_tlwh3, cur_id3):
            x, y, w, h = tlwh
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            point_set3.append((center_x, center_y))
            id_set3.append(id)
        track_len3.append(len(cur_id3))
        if frame_id >= max_trajectory_len:
            remove_len3 = track_len3[frame_id - max_trajectory_len]
            point_set3 = point_set3[remove_len3:]
            id_set3 = id_set3[remove_len3:]
        pointed_img3 = plot_center_point(img0, point_set3, id_set3)
        # plot_img(img0, frame_id, [cur_tlwh3, cur_id3, cur_cls3],
        #          save_dir=os.path.join(SAVE_FOLDER, 'result_images', folder_name2))
        plot_img(pointed_img3, frame_id, [cur_tlwh3, cur_id3, cur_cls3],
                 save_dir=os.path.join(SAVE_FOLDER, 'result_images', folder_name3))
        timer.toc()  # end timing this image
        frame_id += 1

    seq_fps.append(frame_id / timer.total_time)  # cal fps for current seq
    timer.clear()  # clear for next seq
    # thirdly, save results
    # every time assign a different name
    save_results(folder_name1, filename, results1)
    save_results(folder_name2, filename, results2)
    save_results(folder_name3, filename, results3)

    seqs.append(str(filename))

    print(f'average fps: {np.mean(seq_fps)}')
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    with open(f'tracker/config_files/mydata.yaml', 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    yaml_dataset_config = cfgs['TRACK_EVAL']  # read yaml file to read TrackEval configs
    # make sure that seqs is same as 'SEQ_INFO' in yaml
    # delete key in 'SEQ_INFO' which is not in seqs
    seqs_in_cfgs = list(yaml_dataset_config['SEQ_INFO'].keys())
    for k in seqs_in_cfgs:
        if k not in seqs:
            yaml_dataset_config['SEQ_INFO'].pop(k)
    # assert len(yaml_dataset_config['SEQ_INFO'].keys()) == len(seqs)

    for k in default_dataset_config.keys():
        if k in yaml_dataset_config.keys():  # if the key need to be modified
            default_dataset_config[k] = yaml_dataset_config[k]

    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.VisDrone2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity,
                   trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)


class my_queue:
    """
    implement a queue for image seq reading
    """

    def __init__(self, arr: list) -> None:
        self.arr = arr
        self.start_idx = 0

    def push_back(self, item):
        self.arr.append(item)

    def pop_front(self):
        ret = cv2.imread(self.arr[self.start_idx])
        self.start_idx += 1
        return not self.is_empty(), ret

    def is_empty(self):
        return self.start_idx == len(self.arr)


def resize_a_frame(frame, target_size):
    """
    resize a frame to target size

    frame: np.ndarray, shape (H, W, C)
    target_size: List[int, int] | Tuple[int, int]
    """
    # resize to input to the YOLO net
    frame_resized = cv2.resize(frame, (target_size[0], target_size[1]))  # (H', W', C)
    # convert BGR to RGB and to (C, H, W)
    frame_resized = frame_resized[:, :, ::-1].transpose(2, 0, 1)
    frame_resized = np.ascontiguousarray(frame_resized, dtype=np.float32)
    frame_resized /= 255.0

    frame_resized = torch.from_numpy(frame_resized)

    return frame_resized


def save_results(folder_name, filename, results, data_type='default'):
    """
    write results to txt file

    results: list  row format: frame id, target id, box coordinate, class(optional)
    to_file: file path(optional)
    data_type: write data format
    """
    assert len(results)
    if not data_type == 'default':
        raise NotImplementedError  # TODO

    save_dir = os.path.join(SAVE_FOLDER, str(folder_name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(SAVE_FOLDER, str(folder_name), str(filename) + '.txt'), 'w') as f:
        for frame_id, target_ids, tlwhs, clses in results:
            if data_type == 'default':

                # f.write(f'{frame_id},{target_id},{tlwh[0]},{tlwh[1]},\
                #             {tlwh[2]},{tlwh[3]},{cls}\n')
                for id, tlwh, cls in zip(target_ids, tlwhs, clses):
                    f.write(f'{frame_id},{id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{int(cls)}\n')
    f.close()


def plot_img(img, frame_id, results, save_dir):
    """
    img: np.ndarray: (H, W, C)
    frame_id: int
    results: [tlwhs, ids, clses]
    save_dir: sr

    plot images with bboxes of a seq
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_ = np.ascontiguousarray(np.copy(img))

    tlwhs, ids, clses = results[0], results[1], results[2]
    for tlwh, id, cls in zip(tlwhs, ids, clses):
        # convert tlwh to tlbr
        tlbr = tuple([int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])])
        # draw a rect
        cv2.rectangle(img_, tlbr[:2], tlbr[2:], get_color(id), thickness=3, )
        # note the id and cls
        text = f'{CATEGORY_DICT[cls]}-{id}'
        cv2.putText(img_, text, (tlbr[0], tlbr[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                    color=(255, 164, 0), thickness=2)

    cv2.imwrite(filename=os.path.join(save_dir, f'{frame_id:05d}.jpg'), img=img_)


def save_videos(obj_name):
    """
    convert imgs to a video

    seq_names: List[str] or str, seqs that will be generated
    """
    if not isinstance(obj_name, list):
        obj_name = [obj_name]

    for seq in obj_name:
        images_path = os.path.join(SAVE_FOLDER, 'reuslt_images', seq)
        images_name = sorted(os.listdir(images_path))

        to_video_path = os.path.join(images_path, '../../', seq + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        img0 = Image.open(os.path.join(images_path, images_name[0]))
        vw = cv2.VideoWriter(to_video_path, fourcc, 15, img0.size)

        for img in images_name:
            if img.endswith('.jpg'):
                frame = cv2.imread(os.path.join(images_path, img))
                vw.write(frame)

    print('Save videos Done!!')


def get_color(idx):
    """
    aux func for plot_seq
    get a unique color for each id
    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_center_point(img0, point_set, id_set):
    pointed_img = img0.copy()
    i = 0
    for point in point_set:
        x, y = point
        cv2.circle(pointed_img, (x, y), radius=4, color=get_color(id_set[i]), thickness=-1)
        i += 1
    return pointed_img
