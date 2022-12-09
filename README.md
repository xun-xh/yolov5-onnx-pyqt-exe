# Based on Yolov5 + onnx + PyQt5 + nuitka

---
English | [简体中文](.github/README_CN.md)

## \***If you encounter an unsolvable problem, [issue](https://github.com/xun-xh/yolov5-onnx-pyqt-exe/issues/new/choose) are allowed**\*

## \***Welcome to star**\*

---

### Implementation

> - Train custom dataset with yolov5
> - export pt format to onnx format
> - inference with openCV.dnn or onnxruntime
> - Package as executable program on Windows platform (Linux can also be packaged in theory, but it has not been tried)
> - Portable (deployed) to most Windows devices after packaging

---

### Preview

#### download the demo：<https://download.kstore.space/download/3190/plugin/publish.zip>

> ![main window](https://img-blog.csdnimg.cn/a52cbae15c7c4fc19ce5476b6374605f.png)

#### **Function**
>
> 1. support image, video, webcam
> 2. real time frame rate
> 3. redirect stdout to GUI
> 4. change conf_thres and iou_thres at any time
> 5. display/no bounding box, change bounding box's color
> 6. print/hide inference result
> 7. record video
> 8. save screenshot or log
> 9. costom script, triggered on each picture(see the `need/self_demo.py`)
>
> ![function](https://img-blog.csdnimg.cn/93bfdb8ebb844f78b1fb36745d4188a4.png#pic_center)
> ![img_2.png](https://img-blog.csdnimg.cn/d2651fe582694c40b818a798aeb154b6.png#pic_center)

---

### Requirement

> - python >= 3.9
> - numpy == 1.23.4
> - opencv-python == 4.5.5.62
> - PyQt5 == 5.15.7
> - onnxruntime == 1.13.1
> - nuitka == 0.6.18.4

---

### Tutorials

> #### Quickstart
>
> - Clone this Repo
> - Install requirements, `pip install -r requirements.txt`
> - Run `Yolo2onnxDetectProjectDemo.py`
> - Then you will see the GUI
> - Click `启动检测`
>
>#### Train costom data and export to onnx model
>
> - [Yolov5 v5.0](https://github.com/ultralytics/yolov5/tree/v5.0) is recommended
> - How to train?[目标检测---教你利用yolov5训练自己的目标检测模型](https://blog.csdn.net/jiaoty19/article/details/125614783)
> - [Export](https://github.com/ultralytics/yolov5/issues/251) to onnx format: `python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1`
>
>#### Packaging an executable file
>
> - The module used is nuitka, and the command has been configured in `build. py`. If you need more advanced playing methods, please see [https://nuitka.net/](https://nuitka.net/)
> - Run `build.py`, finished products are located in `build_file/publish` folder
>   1. Tips:the truly finished products are in the `Yolo2onnxDetectProjectDemo.dist` folder
>   2. To facilitate debugging and updating, all files in this folder need to be copied to the `publish` folder after the first packaging is successful
>   3. Double click to run the exe file, and copy the module to the `publish` folder according to the exception,  until it runs successfully

---
