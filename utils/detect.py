# -*coding=utf-8

import os
import threading
import time
from typing import Union

import cv2
import numpy


class YOLOv5(object):
    """使用yolo的.pt格式模型转换为.onnx格式模型进行目标识别"""

    def __init__(self, **kwargs):
        self.initConfig(**kwargs)

    def initModel(self, path, t: str = None) -> None:
        """
        初始化模型
        :param path: model path
        :param t: onnxruntime or cv2.dnn, default: onnxruntime
        :return:
        """
        self.t = t
        if t == 'cv2.dnn':
            self.net = cv2.dnn.readNet(path)

            self.output_names = self.net.getUnconnectedOutLayersNames()
        else:
            import onnxruntime
            self.net = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            model_inputs = self.net.get_inputs()
            self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

            model_outputs = self.net.get_outputs()
            self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        self.has_postprocess = 'score' in self.output_names

    def initConfig(self, input_width=640, input_height=640, conf_thres=0.5, iou_thres=0.5, draw_box=True,
                   thickness=2, class_names: Union[tuple, list] = None, box_color=(255, 0, 0),
                   txt_color=(0, 255, 255), with_pos=False,
                   **kwargs) -> None:
        """
        初始化模型配置
        """
        assert 0 < conf_thres <= 1 and 0 < iou_thres <= 1
        self.input_width = input_width  # 输入图片宽
        self.input_height = input_height  # 输入图片高
        self.conf_threshold = conf_thres  # 置信度
        self.iou_threshold = iou_thres  # IOU
        self.draw_box = draw_box  # 是否画锚框
        self.thickness = thickness  # 锚框宽度
        self.class_names = class_names  # 类别
        self.box_color = box_color  # 锚框颜色 BGR
        self.txt_color = txt_color  # 文字颜色 BGR
        self.with_pos = with_pos  # 是否返回坐标
        self.__dict__.update(kwargs)

    def detect(self, image: numpy.ndarray) -> dict:
        outputs = self.__inference(image)
        boxes, scores, class_ids = self.__postProcess(outputs)
        res_ = self.__formatResult(image, boxes, scores, class_ids)
        return res_

    def __prepareInput(self, image):
        self.img_height, self.img_width = image.shape[:2]

        # Resize input image
        input_img = cv2.resize(image, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[numpy.newaxis, :, :, :].astype(numpy.float32)
        return input_tensor

    def __inference(self, image: numpy.ndarray) -> list[numpy.ndarray]:
        """
        :param image: 待检测图像 RGB格式
        :return:
        """
        input_tensor = self.__prepareInput(image)
        # blob = cv2.dnn.blobFromImage(input_img, 1 / 255.0)
        # Perform inference on the image
        if self.t == 'cv2.dnn':
            self.net.setInput(input_tensor)
            # Runs the forward pass to get output of the output layers
            outputs = self.net.forward(self.output_names)
        else:
            outputs = self.net.run(self.output_names, {self.input_names[0]: input_tensor})
        # print(outputs[0].shape)  # (1, 25200, 85)
        return outputs

    def __postProcess(self, outputs):
        if self.has_postprocess:
            res_ = self.__parseProcessedOutput(outputs)
        else:
            # Process output data
            res_ = self.__processOutput(outputs)
        return res_

    def __processOutput(self, output) -> tuple:
        predictions = numpy.squeeze(output[0])

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, numpy.newaxis]

        # Get the scores
        scores = numpy.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores]
        scores = scores[valid_scores]

        # Get the class with the highest confidence
        class_ids = numpy.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self.__extractBoxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold)
        indices = numpy.array(indices)
        indices = indices.flatten()
        if indices.size > 0:
            return boxes[indices], scores[indices], class_ids[indices]
        else:
            return None, None, None

    def __parseProcessedOutput(self, outputs):
        scores = numpy.squeeze(outputs[self.output_names.index('score')])
        predictions = outputs[self.output_names.index('batchno_classid_x1y1x2y2')]

        # Filter out object scores below threshold
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores, :]
        scores = scores[valid_scores]

        # Extract the boxes and class ids
        # batch_number = predictions[:, 0]
        class_ids = predictions[:, 1]
        boxes = predictions[:, 2:]

        # In postprocess, the x,y are the y,x
        boxes = boxes[:, [1, 0, 3, 2]]

        # Rescale boxes to original image dimensions
        boxes = self.__rescaleBoxes(boxes)

        return boxes, scores, class_ids

    def __extractBoxes(self, predictions):
        boxes = predictions[:, :4]  # Extract boxes from predictions
        boxes = self.__rescaleBoxes(boxes)  # Scale boxes to original image dimensions
        # Convert boxes to xyxy format
        boxes_ = numpy.copy(boxes)
        boxes_[..., 0] = boxes[..., 0] - boxes[..., 2] * 0.5
        boxes_[..., 1] = boxes[..., 1] - boxes[..., 3] * 0.5
        boxes_[..., 2] = boxes[..., 0] + boxes[..., 2] * 0.5
        boxes_[..., 3] = boxes[..., 1] + boxes[..., 3] * 0.5

        return boxes_

    def __rescaleBoxes(self, boxes):
        """Rescale boxes to original image dimensions"""
        input_shape = numpy.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = numpy.divide(boxes, input_shape, dtype=numpy.float32)
        boxes *= numpy.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def __formatResult(self, image, boxes, scores, class_ids) -> dict:
        """
        格式化检测结果
        :param image: 输入图像
        :param boxes:
        :param scores:
        :param class_ids:
        :return: 检测结果
        """
        detection = {}
        if boxes is None:
            return detection
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            label = self.class_names[class_id]
            if label == '_':
                continue
            if label in detection:
                detection[label]['num'] += 1
                detection[label]['score'].append(score)
                if self.with_pos: detection[label]['pos'].append((x1, y1, x2, y2))
            else:
                detection[label] = {'num': 1, 'score': [score, ], }
                if self.with_pos: detection[label].update({'pos': [(x1, y1, x2, y2), ]})

            if self.draw_box:
                label_p = f'{label} {int(score * 100)}%'
                lw = max(round(sum(image.shape) / 2 * 0.003), self.thickness)  # line width
                cv2.rectangle(image, (x1, y1), (x2, y2), self.box_color, thickness=lw, lineType=cv2.LINE_AA)
                if label_p:
                    tf = max(lw - 1, 1)  # font thickness
                    w, h = cv2.getTextSize(label_p, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                    outside = y1 - h >= 3
                    p2 = x1 + w, y1 - h - 3 if outside else y1 + h + 3
                    cv2.rectangle(image, (x1, y1), p2, self.box_color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(image,
                                label_p, (x1, y1 - 2 if outside else y1 + h + 2),
                                0,
                                lw / 3,
                                self.txt_color,
                                thickness=tf,
                                lineType=cv2.LINE_AA)
        return detection


class DataLoader(object):
    """逐帧加载图像，返回RGB格式"""
    VIDEO_TYPE = ('asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv')
    IMAGE_TYPE = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm')

    def __init__(self, path: Union[int, str], frame_skip=-1):
        """
        :param path: image/video path
        :param frame_skip: frame skip or not, <0: auto; =0: dont skip; >0: skip  # video only
        """
        self.isFinished = False
        self.path = path
        self.is_wabcam_0 = path == 0
        self.is_wabcam_1 = path == 1
        self.is_video = isinstance(path, str) and path.endswith(DataLoader.VIDEO_TYPE)
        self.is_image = isinstance(path, str) and path.endswith(DataLoader.IMAGE_TYPE)
        assert self.is_wabcam_0 or self.is_wabcam_1 or self.is_video or self.is_image, '?'

        if self.is_wabcam_0 or self.is_wabcam_1:
            self.cap = cv2.VideoCapture(path, cv2.CAP_DSHOW)
            self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        elif self.is_video:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            self.frame_skip = frame_skip
            self.idx = 0
            if frame_skip < 0:
                class VideoFrameDraw(threading.Thread):
                    def __init__(self):
                        super(VideoFrameDraw, self).__init__(daemon=True)
                        self.cap = cv2.VideoCapture(path)
                        self.grab = self.cap.grab
                        self.isOpened = self.cap.isOpened
                        self.release = self.cap.release

                        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        self.w, self.h = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        self.ret = self.cap.grab() if self.cap.isOpened() else False
                        if not self.ret:
                            return
                        self.ret, self.img = self.cap.retrieve()
                        if not self.ret or self.img is None:
                            return
                        self.pause = threading.Event()
                        self.read_frame = False

                        self.start()

                    def retrieve(self):
                        if not self.pause.isSet(): self.pause.set()
                        if self.ret:
                            self.read_frame = True
                            return self.ret, self.img
                        return False, None

                    def run(self) -> None:
                        while self.cap.isOpened():
                            self.ret = self.cap.grab()
                            self.pause.wait()
                            if not self.ret:
                                break
                            time.sleep(self.fps / 1000)
                            if not self.read_frame:
                                continue
                            self.ret, self.img = self.cap.retrieve()
                            self.read_frame = False
                            if not self.ret:
                                break

                    def __del__(self):
                        self.release()

                self.cap = VideoFrameDraw()
                self.w, self.h = int(self.cap.w), int(self.cap.h)
                self.fps = self.cap.fps
            else:  # frame_skip >= 0
                self.cap = cv2.VideoCapture(path)
                self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        elif self.is_image:
            if not os.path.exists(path):
                raise FileNotFoundError(path)

    def __next__(self) -> tuple[numpy.ndarray, str]:
        if (self.is_wabcam_0 or self.is_wabcam_1) and self.cap.isOpened():
            ret, img = self.cap.read()
            if self.is_wabcam_0: img = cv2.flip(img, 1)
            path = ''
        elif self.is_video and self.cap.isOpened():
            while self.idx <= self.frame_skip:
                ret = self.cap.grab()
                self.idx += 1
                if not ret:
                    raise StopIteration
            ret, img = self.cap.retrieve()
            self.idx = 0
            path = self.path
        elif self.is_image and not self.isFinished:
            self.isFinished = True
            ret, img = True, cv2.imread(self.path)
            path = self.path
        else:
            raise StopIteration

        if not ret or img is None:
            raise StopIteration
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, os.path.basename(path)

    def __iter__(self):
        return self

    def __del__(self):
        if 'cap' in self.__dict__:
            self.cap.release()

