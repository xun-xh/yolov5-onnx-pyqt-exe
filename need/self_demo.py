"""
自定义脚本,
每次检测都将触发执行此文件内的代码, 并提供名为"data_api"的接口, 使用print(data_api.help)来获取帮助
阻塞主线程,
自动import自定义脚本文件根目录下的模块和包, 也可以自行sys.path.append,
此文件位于./need/self_demo.py
"""


# ----------
# 若未定义data_api, 则先使用假数据初始化data_api, 供测试用, 可删除此部分
if 'data_api' not in locals().keys():
    class API(object):
        pass
    data_api = API()
    data_api.res_data = {'person': {'num': 2, 'score': [0.93594641, 0.85145649, ]},
                         'bottle': {'num': 1, 'score': [0.79432157, ]}}


# ----------
# 自定义脚本示例1.
def myPrint(*sth):
    print(*sth)

if 'person' in data_api.res_data:
    myPrint(f"{data_api.res_data['person']['num']} person in the picture now")


# ----------
# 自定义脚本示例2.
import threading, time

class ExampleThread(threading.Thread):
    def __init__(self):
        super().__init__(name='example_thread', daemon=True)

    def run(self):
        print(data_api.res_data)
        time.sleep(10)  # 在此处理耗时长的逻辑而不阻塞主线程

def startThread():
    for i in threading.enumerate():
        if i.name == 'example_thread':
            return
    ExampleThread().start()

startThread()
