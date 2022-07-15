import argparse
import norfair
import numpy as np
import torch
import yolov5
from typing import Union, List, Optional

import norfair
from norfair import Detection, Tracker, Video

import cv2
from scipy.spatial import distance
from itertools import combinations

max_distance_between_points: int = 30
max_distance_between_people: int = 110

class YOLO:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # load model
        self.model = yolov5.load(model_path, device=device)

    def __call__(
            self,
            img: Union[str, np.ndarray],
            conf_threshold: float = 0.25,
            iou_threshold: float = 0.45,
            image_size: int = 720,
            classes: Optional[List[int]] = None
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def yolo_detections_to_norfair_detections(
        yolo_detections: torch.tensor,
        track_points: str = 'centroid'  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections
    """
    norfair_detections: List[Detection] = []

    if track_points == 'centroid':
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [
                    detection_as_xywh[0].item(),
                    detection_as_xywh[1].item()
                ]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(points=centroid, scores=scores)
            )
    elif track_points == 'bbox':
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()]
                ]
            )
            scores = np.array([detection_as_xyxy[4].item(), detection_as_xyxy[4].item()])
            norfair_detections.append(
                Detection(points=bbox, scores=scores)
            )

    return norfair_detections

def violenceboxdraw(img, pts_list):
        n = 0
        violation_pts_x = [ ]
        violation_pts_y = [ ]
        
        points = [(x,y) for (x,y) in pts_list]
        
        points_combo = list(combinations(points, 2))
        
        for i in points_combo:
            dist = distance.euclidean(i[0],i[1])
            
            if dist <= max_distance_between_people:
                n += 1
                violation_pts_x.append(i[0][0])
                violation_pts_x.append(i[1][0])
                violation_pts_y.append(i[0][1])
                violation_pts_y.append(i[1][1])
        
        min_x, min_y = int(min(violation_pts_x)), int(min(violation_pts_y))
        max_x, max_y = int(max(violation_pts_x)), int(max(violation_pts_y))

        if n >= 4:
            #print('Violation Detected!')
            cv2.putText(img, 'SECTION 144 VIOLATED!', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3)
            # box_drawn_img = cv2.rectangle(img, (10,10), (int(video.input_width)-10, int(video.input_height)-10), (0,0,255), 2)
            box_drawn_img = cv2.rectangle(img, (min_x, min_y), (max_x,max_y), (0,0,255), 2)
            return box_drawn_img
        else:
            return img



parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument("files", type=str, nargs="+", help="Video files to process")
parser.add_argument("--detector_path", type=str, default="yolov5m6.pt", help="YOLOv5 model path")
parser.add_argument("--img_size", type=int, default="720", help="YOLOv5 inference size (pixels)")
parser.add_argument("--conf_thres", type=float, default="0.25", help="YOLOv5 object confidence threshold")
parser.add_argument("--iou_thresh", type=float, default="0.45", help="YOLOv5 IOU threshold for NMS")
parser.add_argument('--classes', nargs='+', type=int, help='Filter by class: --classes 0, or --classes 0 2 3')
parser.add_argument("--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'")
parser.add_argument("--track_points", type=str, default="centroid", help="Track points: 'centroid' or 'bbox'")
args = parser.parse_args()

model = YOLO(args.detector_path, device=args.device)

for input_path in args.files:
    video = Video(input_path=input_path)
    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=max_distance_between_points,
    )

    for frame in video:
        yolo_detections = model(
            frame,
            conf_threshold=args.conf_thres,
            iou_threshold=args.conf_thres,
            image_size=args.img_size,
            classes=args.classes
        )
        #print(type(yolo_detections), yolo_detections.xywh)
        predictions = yolo_detections.pred[0]
        boxes = predictions[:, :4]  # x1, x2, y1, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]
        #print(type(boxes))

        ###########################get the cordinates of bounding boxes or centroids , scores are extracted but not added to the list co_list_ob ################################
        co_list_ob = []
        if args.track_points == 'centroid':
            detections_as_xywh = yolo_detections.xywh[0]
            for detection_as_xywh in detections_as_xywh:
                centroid = np.array(
                    [
                        detection_as_xywh[0].item(),
                        detection_as_xywh[1].item()
                    ]
                )
                scores = np.array([detection_as_xywh[4].item()])
                co_list_ob.append(
                    centroid
                )
        elif args.track_points == 'bbox':
            detections_as_xyxy = yolo_detections.xyxy[0]
            for detection_as_xyxy in detections_as_xyxy:
                bbox = np.array(
                    [
                        [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                        [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()]
                    ]
                )
                scores = np.array([detection_as_xyxy[4].item(), detection_as_xyxy[4].item()])
                co_list_ob.append(
                    bbox
                )
        #print("object list",co_list_ob)




        ############################################################
        detections = yolo_detections_to_norfair_detections(yolo_detections, track_points=args.track_points)

        tracked_objects = tracker.update(detections=detections)
        if args.track_points == 'centroid':
            norfair.draw_points(frame, detections)
        elif args.track_points == 'bbox':
            norfair.draw_boxes(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)

        if cv2.waitKey(1) == ord('q') : # Press 'q' to quit the code
            exit()

        frame = violenceboxdraw(frame, co_list_ob)  # Violation Detect function

        #frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
        video.write(frame)
