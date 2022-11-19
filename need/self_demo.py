"""
自定义脚本,
每次检测都将触发执行此文件内的代码, 并提供名为"api_data"的接口,
接口已内置：
api_data.res_data: dict类型 检测结果,
api_data.img_data: numpy.ndarray类型 实时图像,
阻塞主线程,
自动import自定义脚本文件根目录下的模块和包, 也可以自行sys.path.append,
此文件位于./need/self_demo.py
"""


# ----------
# 若未定义api_data, 则先使用假数据初始化api_data, 供测试用, 可删除此部分
if 'api_data' not in locals().keys():
    api_data.res_data = {'person': {'num': 2, 'score': [0.93594641, 0.85145649, ]},
                         'bottle': {'num': 1, 'score': [0.79432157, ]}}


# ----------
# 自定义脚本示例1.
def myPrint(*sth):
    print(*sth)

if 'person' in api_data.res_data:
    myPrint(f"[self_demo.py]: {api_data.res_data['person']['num']}")


# ----------
# 自定义脚本示例2.
import threading, time

class MyThread(threading.Thread):
    def __init__(self):
        super().__init__(name='example_thread', daemon=True)

    def run(self):
        print(f"[self_demo.py]: {api_data.res_data}")
        time.sleep(10)  # 在此处理耗时长的逻辑而不阻塞主线程

def startThread():
    for i in threading.enumerate():
        if i.name == 'example_thread':
            return
    MyThread().start()

startThread()
