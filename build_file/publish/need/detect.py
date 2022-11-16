import logging
import os
import threading
import time
from typing import Union

import cv2
import numpy
from PyQt5 import QtCore

import onnxruntime as ort


class YOLO:
    """使用yolo的.pt格式模型转换为.onnx格式模型进行目标识别"""

    def __init__(self):
        self.initConfig()

    def initModel(self, path, t: str = None):
        # Initialize model
        self.t = t
        if t == 'cv2.dnn':
            self.net = cv2.dnn.readNet(path)

            self.output_names = self.net.getUnconnectedOutLayersNames()
        else:
            self.net = ort.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            model_inputs = self.net.get_inputs()
            self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

            model_outputs = self.net.get_outputs()
            self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        self.has_postprocess = 'score' in self.output_names

    def initConfig(self, input_width=640, input_height=480, conf_thres=0.7, iou_thres=0.5, class_names: tuple = None,
                   draw_box=True, box_color=(255, 0, 0), thickness=2, **kwargs):
        self.input_width = input_width  # 输入图片宽
        self.input_height = input_height  # 输入图片高
        self.conf_threshold = conf_thres  # 置信度
        self.iou_threshold = iou_thres  # IOU
        self.class_names = class_names  # 类别
        self.draw_box = draw_box  # 是否画锚框
        self.box_color = box_color[::-1]  # 锚框颜色  RGB
        self.thickness = thickness  # 锚框宽度

    def detect(self, image: numpy.ndarray):
        """
        :param image: 待检测图像
        :return: boxes位置, scores置信度, class_ids类别id
        """
        input_tensor = self.prepare_input(image)
        # blob = cv2.dnn.blobFromImage(input_img, 1 / 255.0)
        # Perform inference on the image
        if self.t == 'cv2.dnn':
            self.net.setInput(input_tensor)
            # Runs the forward pass to get output of the output layers
            outputs = self.net.forward(self.output_names)
        else:
            outputs = self.net.run(self.output_names, {self.input_names[0]: input_tensor})

        if self.has_postprocess:
            _res = self.parse_processed_output(outputs)
        else:
            # Process output data
            _res = self.process_output(outputs)

        if _res is not None:
            return _res
        return None, None, None

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[numpy.newaxis, :, :, :].astype(numpy.float32)
        return input_tensor

    def process_output(self, output):
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
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold)
        indices = numpy.array(indices)
        indices = indices.flatten()
        if indices.size > 0:
            return boxes[indices], scores[indices], class_ids[indices]

    def parse_processed_output(self, outputs):

        scores = numpy.squeeze(outputs[self.output_names.index('score')])
        predictions = outputs[self.output_names.index('batchno_classid_x1y1x2y2')]

        # Filter out object scores below threshold
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores, :]
        scores = scores[valid_scores]

        # Extract the boxes and class ids
        # TODO: Separate based on batch number
        # batch_number = predictions[:, 0]
        class_ids = predictions[:, 1]
        boxes = predictions[:, 2:]

        # In postprocess, the x,y are the y,x
        boxes = boxes[:, [1, 0, 3, 2]]

        # Rescale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        return boxes, scores, class_ids

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes_ = numpy.copy(boxes)
        boxes_[..., 0] = boxes[..., 0] - boxes[..., 2] * 0.5
        boxes_[..., 1] = boxes[..., 1] - boxes[..., 3] * 0.5
        boxes_[..., 2] = boxes[..., 0] + boxes[..., 2] * 0.5
        boxes_[..., 3] = boxes[..., 1] + boxes[..., 3] * 0.5

        return boxes_

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = numpy.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = numpy.divide(boxes, input_shape, dtype=numpy.float32)
        boxes *= numpy.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, boxes, scores, class_ids):
        """
        画锚框
        :param image: 输入图像
        :param boxes: from self.detect()
        :param scores: from self.detect()
        :param class_ids: from self.detect()
        :return: 检测结果, 输出图像
        """
        res = {}
        if boxes is None:
            return res, image
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)

            label = self.class_names[class_id]
            if label in res:
                res[label]['num'] += 1
                res[label]['score'].append(score)
            else:
                res[label] = {'num': 1, 'score': [score, ]}
            label = f'{label} {int(score * 100)}%'
            if self.draw_box:
                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), color=self.box_color, thickness=self.thickness)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return res, image


class DataLoader(object):
    Video_Type = ('asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv')
    Image_Type = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm')

    def __init__(self, path: Union[int, str], frame_draw=True):
        self.path = path
        self.is_front_wabcam = path == 0
        self.is_back_wabcam = path == 1
        self.is_video = isinstance(path, str) and path.endswith(DataLoader.Video_Type)
        self.is_image = isinstance(path, str) and path.endswith(DataLoader.Image_Type)
        assert self.is_front_wabcam or self.is_back_wabcam or self.is_video or self.is_image, '?'

        if self.is_front_wabcam or self.is_back_wabcam:
            self.cap = cv2.VideoCapture(path, cv2.CAP_DSHOW)
        elif self.is_video:
            if not os.path.exists(path):
                raise FileNotFoundError
            if frame_draw:
                class VideoFrameDraw(threading.Thread):
                    def __init__(self):
                        super(VideoFrameDraw, self).__init__(daemon=True)
                        self.cap = cv2.VideoCapture(path)
                        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                        self.frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        self.ret, self.img = self.cap.read()
                        self.pause = threading.Event()

                    def isOpened(self):
                        return self.cap.isOpened()

                    def read(self):
                        self.pause.set()
                        return self.ret, self.img

                    def release(self):
                        self.cap.release()

                    def run(self) -> None:
                        while self.cap.isOpened():
                            ret, img = self.cap.read()
                            self.ret = ret
                            if ret:
                                self.pause.wait()
                                self.img = img
                                time.sleep(self.fps / 1000)
                            else:
                                break

                    def __del__(self):
                        self.cap.release()

                self.cap = VideoFrameDraw()
                self.cap.start()
            else:
                self.cap = cv2.VideoCapture(path)
        elif self.is_image:
            if not os.path.exists(path):
                raise FileNotFoundError

    def __next__(self) -> tuple[numpy.ndarray, str]:
        if self.is_front_wabcam:
            ret, img = self.cap.read()
            if not ret:
                raise StopIteration
            img = cv2.flip(img, 1)
            return img, ''
        elif self.is_back_wabcam:
            ret, img = self.cap.read()
            if not ret:
                raise StopIteration
            return img, ''
        elif self.is_video:
            ret, img = self.cap.read()
            if not ret:
                raise StopIteration
            return img, self.path
        elif self.is_image:
            return cv2.imread(self.path), self.path

    def __iter__(self):
        return self

    def __del__(self):
        if self.is_front_wabcam or self.is_back_wabcam or self.is_video:
            self.cap.release()


class DetectThread(QtCore.QThread):
    """检测线程"""
    img_sig = QtCore.pyqtSignal(numpy.ndarray)
    res_sig = QtCore.pyqtSignal(dict)

    def __init__(self, model: YOLO = None, dataset: DataLoader = None):
        super(DetectThread, self).__init__()
        self.is_running = True
        self.is_detecting = False
        self.model = model
        self.dataset: DataLoader = dataset
        self.display_fps = True
        self.print_result = True

    def stopThread(self):
        self.is_running = False
        self.is_detecting = False

    def stopDetect(self):
        self.is_detecting = False

    def startThread(self):
        self.is_detecting = False
        self.is_running = True
        if not self.isRunning():
            self.start()

    def startDetect(self):
        self.is_detecting = True
        self.is_running = True
        if not self.isRunning():
            self.start()

    def main(self):  # 主函数
        fps = '--'
        fps_count = 0
        t = time.time()
        for img, path in self.dataset:
            if not self.is_running:
                break
            res = {}
            if self.is_detecting:
                try:
                    # boxes, scores, class_ids = self.model.detect(img)
                    res, _ = self.model.draw_detections(img, *self.model.detect(img))
                    if self.print_result:  # 打印结果
                        print(res)
                except Exception as e:
                    print(e)
                    self.is_detecting = False
                    print('stop')

            # 显示帧数
            ted = time.time() - t
            if ted >= 2:
                fps = '%.1f' % (fps_count / ted)
                fps_count = 0
                t = time.time()
            else:
                fps_count += 1
            if self.display_fps:
                cv2.putText(img, f'FPS:{fps}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

            self.img_sig.emit(img)
            self.res_sig.emit(res)
            time.sleep(0.0001)

        del self.dataset
        print('exit')
        self.is_detecting = False

    def run(self) -> None:
        try:
            self.main()
        except Exception as e:
            logging.exception(e)
