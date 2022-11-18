import argparse
import cgitb
import logging
import os
import sys

from PyQt5 import QtWidgets, QtCore
import qt_material  # 一个美化pyqt界面的第三方库

from utils.Main_Window import MainWindow, StdOut

if __name__ == "__main__":
    # todo:命令行模式
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='need/models/yolov7-tiny_640x640.onnx')
    parser.add_argument('--class', type=str, default='need/coco_class.txt')
    parser.add_argument('--source', type=str)
    parser.add_argument('--conf-thres', type=float, default=0.5)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    opt = parser.parse_args()

    # dump logs
    log_dir = os.path.join(os.getcwd(), 'log')
    os.makedirs(log_dir, exist_ok=True)
    cgitb.enable(format='text', logdir=log_dir)

    # init GUI
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 高分辨率
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = MainWindow()
    # qt_material.apply_stylesheet(app, 'dark_blue.xml')  # 界面美化

    # redirect stdout
    stdout = StdOut()
    stdout.signalForText.connect(mainwindow.displayLog)
    sys.stdout = stdout
    sys.stderr = stdout
    logging.StreamHandler(stdout)

    # show main window
    mainwindow.show()
    sys.exit(app.exec_())
