"""
自定义脚本
每次检测都将触发执行此文件内的代码, 并将检测结果以"res_data"作为变量名传入, 类型为字典
阻塞主线程
自动import自定义脚本文件根目录下的模块和包, 也可以自行os.path.append
此文件位于./need/self_demo.py
"""


# ----------
# 若未定义res_data, 则先使用假数据初始化res_data, 供测试用, 可删除此部分
if 'res_data' not in locals().keys():
    res_data = {'person': {'num': 2, 'score': [0.93594641, 0.85145649, ]},
                'bottle': {'num': 1, 'score': [0.79432157, ]}}


# ----------
# 自定义脚本示例1.
def myPrint(*sth):
    print(*sth)

if 'person' in res_data:
    myPrint(f"[self_demo.py]: {res_data['person']['num']}")


# ----------
# 自定义脚本示例2.
import threading, time

class MyThread(threading.Thread):
    def __init__(self):
        super().__init__(name='example_thread', daemon=True)

    def run(self):
        print(f"[self_demo.py]: {res_data}")
        time.sleep(10)  # 在此处理耗时长的逻辑而不阻塞主线程

def startThread():
    for i in threading.enumerate():
        if i.name == 'example_thread':
            return
    MyThread().start()

startThread()
