# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.
## create file with sample code
import torch
import numpy as np
from azure.iot.device.aio import IoTHubModuleClient
from azure.iot.device import Message
from datetime import datetime
import asyncio
import paho.mqtt.client as mqtt #import the client1
import time 
import paho.mqtt.publish as publish #import the client1
import argparse
import sys
from pathlib import Path
#import onnx
#import onnxruntime
import cv2
import torch.backends.cudnn as cudnn
from utils.tracker.centroidtracker import CentroidTracker
from utils.tracker.trackableobject import TrackableObject
import numpy as np
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


broker_address="192.168.0.246" 

def generateCentroid(rects):
    inputCentroids = np.zeros((len(rects), 2), dtype="int")
    for (i, (startX, startY, endX, endY)) in enumerate(rects):
        cX = int((startX + endX) / 2.0)
        cY = int((startY + endY) / 2.0)
        inputCentroids[i] = (cX, cY)
    return inputCentroids

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt




async def startFaceDetectionService():
    # The client object is used to interact with your Azure IoT Edge device.
    module_client = IoTHubModuleClient.create_from_connection_string("HostName=testhub321.azure-devices.net;DeviceId=sensoredge;SharedAccessKey=YId4cq/nZP1CYLgIx0CK3gakA/fnlSftNpJl+/i5TNA=", websockets=True)
    
    ########################### START INSERT ########################################################


@torch.no_grad()
def run(weights='savedModels/yolov5s.pt',  # model.pt path(s)
        source=0, 
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    print(names)
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    ct = CentroidTracker()
    listDet = ['person', 'bicycle']
    totalDownPerson = 0
    totalDownBicycle = 0
    totalUpPerson = 0
    totalUpBicycle = 0
    trackableObjects = {}
    counts_hist =0

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img,
                     augment=augment,
                     visualize=increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        rects = []
        labelObj = []
        yObj = []
        arrCentroid = []

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            height, width, channels = im0.shape

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)
                    x = xyxy
                    tl = None or round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line/font thickness
                    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                    label1 = label.split(' ')
                    if label1[0] in listDet:
                        box = (int(x[0]), int(x[1]), int(x[2]), int(x[3]))
                        rects.append(box)
                        labelObj.append(label1[0])
                        cv2.rectangle(im0, c1, c2, (0, 0, 0), thickness=tl, lineType=cv2.LINE_AA)
                        tf = max(tl - 1, 1)
                        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                        #cv2.rectangle(im0, c1, c2, (0, 100, 0), -1, cv2.LINE_AA)
                        #cv2.putText(im0, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                        #            lineType=cv2.LINE_AA)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    values, counts = np.unique(labelObj, return_counts=True)
                    
                    [print(v + ": " + str(c)) for v, c in zip(values, counts) if values in listDet]
                    if(values.size==0):
                        values=['person']
                        counts=[0]

                    #[print("new person appeared: " + v + ": " + str(c)) for v, c in zip(values, counts) if ((counts > counts_hist) and (values in listDet))]    
                    if (counts > counts_hist):
                        publish.single("test", "New person detected", hostname=broker_address, port=1884) # this is the MQTT breakout
                    
                    counts_hist = counts 

                detCentroid = generateCentroid(rects)
                objects = ct.update(rects)
                for (objectID, centroid) in objects.items():
                    arrCentroid.append(centroid[1])
                for (objectID, centroid) in objects.items():
                    #print(idxDict)
                    to = trackableObjects.get(objectID, None)
                    if to is None:
                        to = TrackableObject(objectID, centroid)
                    else:
                        y = [c[1] for c in to.centroids]
                        direction = centroid[1] - np.mean(y)
                        to.centroids.append(centroid)
                        if not to.counted:  # arah up

                            if direction < 0 and  centroid[1] < height / 1.5 and centroid[
                                1] > height / 1.7:  ##up truble when at distant car counted twice because bbox reappear
                                idx = detCentroid.tolist().index(centroid.tolist())
                                if (labelObj[idx] == 'person'):
                                    totalUpPerson += 1
                                    to.counted = True
                                elif (labelObj[idx] == 'bicycle'):
                                    totalUpBicycle += 1
                                    to.counted = True

                            elif direction > 0 and  centroid[1] > height / 1.5:  # arah down
                                idx = detCentroid.tolist().index(centroid.tolist())
                                if (labelObj[idx] == 'person'):
                                    totalDownPerson += 1
                                    to.counted = True
                                elif (labelObj[idx] == 'bicycle'):
                                    totalDownBicycle += 1
                                    to.counted = True

                    trackableObjects[objectID] = to

            #cv2.putText(im0, 'Down Person : ' + str(totalDownPerson), (int(width * 0.52), int(height * 0.05)),
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 100), 2)
            #cv2.putText(im0, 'Up Person : ' + str(totalUpPerson), (int(width * 0.015), int(height * 0.05)),
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 100), 2)

            # Print time (inference + NMS)
            #print(f'{s}Done. ({t2 - t1:.3f}s)')

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)    
    ########################### END INSERT ##########################################################

    except KeyboardInterrupt:
        print("Process interupted by keyboard interaction")
        msg = Message("Process interupted by keyboard interaction. Module should restart automatically") # thats the payload
        msg.message_id = str(random.randint(0,1000000))
        msg.custom_properties["event_time"] = str(datetime.now())
        await module_client.send_message_to_output(msg, "output1")
        p.terminate()
        await module_client.disconnect()


if __name__ == "__main__":
    opt = parse_opt()
    asyncio.run(startFaceDetectionService(opt))