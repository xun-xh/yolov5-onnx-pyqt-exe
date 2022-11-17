`./need`

## 此文件夹是目标检测相关内容

`models`文件夹内为成品onnx模型，可直接使用。
* `./need/models/helmet_1.onnx`为自己训练的模型，其他两个为官方提供的模型。
* 使用`./need/models/yolov7_480x640.onnx`时需注意，要将输入尺寸改为`640*480`

`coco_class.txt`为Yolo官方训练好的模型的类别文件，内含80种可检测物品

`detect.py`为本项目实际识别时的线程类、检测类、数据类等

`helmet_class.txt`为自己训练的安全帽识别模型的类别文件，内含2种可检测物品

`self_demo.py`为自定义脚本示例文件。


## 关于自定义脚本：
* 若勾选script页内的enable，则每次检测都会触发执行自定义脚本![img.png](https://img-blog.csdnimg.cn/88111b352a864e72860229a0e42097ee.png#pic_center)
* 可以通过此功能实现：若检测到xx，则执行某种命令。
* 例如：如果检测到有人在画面内，就可以使用`pymysql`连接数据库，将时间等数据上传到数据库
* 还可以使用`requests库`post某个地址，等各种骚操作
* 更多文档请参考`need/self_demo.py`