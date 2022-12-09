# 基于Yolov5 + onnx + PyQt5 + 的目标检测打包部署

---
[English](.github/README_en.md) | 简体中文

## \***如遇问题，请提issue**\* \***欢迎star**\*

---

### 实现

> - Yolov5训练个人数据集
> - pt格式模型转换为onnx格式
> - 使用openCV的dnn模块或onnxruntime实现检测
> - 在Windows平台打包为可执行程序(Linux理论上也可以打包，但没试过)
> - 打包后可移植(部署)到大多数Windows设备

---

### 展示

#### 成品下载体验：<https://download.kstore.space/download/3190/plugin/publish.zip>

#### 主界面

> ![主界面](https://img-blog.csdnimg.cn/a52cbae15c7c4fc19ce5476b6374605f.png)

#### **功能**
>
> 1. 支持视频、图片、摄像头
> 2. 实时帧数
> 3. 重定向控制台输出到软件界面上
> 4. 更改检测置信度、IOU阈值
> 5. 显示/关闭锚框、更改锚框宽度及颜色
> 6. 打印/隐藏检测结果
> 7. 录制检测视频
> 8. 保存实时截图、控制台记录
> 9. 自定义脚本，每次检测都将触发，(详细说明请阅读need/self_demo.py)
>
> ![功能](https://img-blog.csdnimg.cn/93bfdb8ebb844f78b1fb36745d4188a4.png#pic_center)
> ![img_2.png](https://img-blog.csdnimg.cn/d2651fe582694c40b818a798aeb154b6.png#pic_center)

---

### 项目需求 (详见requirements.txt)

> - python == 3.9
> - numpy == 1.23.4
> - opencv-python == 4.5.5.62
> - PyQt5 == 5.15.7
> - onnxruntime == 1.13.1
> - nuitka == 0.6.18.4

---

### 使用方法

> #### 快速入门
>
> - clone项目到本地
> - 安装依赖`pip install -r requirements.txt`
> - 运行`Yolo2onnxDetectProjectDemo.py`
> - 如果不报错的话将会出现界面，**有报错又不知道怎么解决的话可以提issue，看到回复**
> - 点击`启动检测`按钮开始检测，高阶玩法参考`need/self_demo.py`
>
>#### 训练自己的数据集并转换为此项目可用的模型
>
> - 推荐用Yolov5 5.0版本，如果想兼容其他版本请自行修改代码，[Yolov5 5.0传送门](https://github.com/ultralytics/yolov5/tree/v5.0)
> - 训练教程：[目标检测---教你利用yolov5训练自己的目标检测模型](https://blog.csdn.net/jiaoty19/article/details/125614783)
> - 训练完成后按照[官方命令](https://github.com/ultralytics/yolov5/issues/251)转为onnx格式。本仓库的模型转换命令为`python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1`
>
>#### 打包为可执行文件
>
> - 所用库为nuitka，打包命令已经在`build.py`中配置好，如需更高级玩法请自己摸索
> - 执行`build.py`，打包好的文件位于`build_file/publish`文件夹
>   1. 此处需注意:真正打包好的文件在`Yolo2onnxDetectProjectDemo.dist`文件夹
>   2. 为了方便debug和更新，在第一次打包成功后需要将此文件夹内所有的文件复制到`publish`文件夹
>   3. 双击运行exe文件，根据报错信息将模块也复制到`publish`文件夹内，直到成功运行
> - 附nuitka的使用方法：[知乎@Python与模具](https://zhuanlan.zhihu.com/p/341099225)

---
