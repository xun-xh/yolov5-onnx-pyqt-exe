import argparse
import cgitb
import logging
import os
import sys

from PyQt5 import QtWidgets, QtCore

from utils.Main_Window import MainWindow, StdOut

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7_480x640.onnx')
    opt = parser.parse_args()

    # dump logs
    log_dir = os.path.join(os.getcwd(), 'log')
    os.makedirs(log_dir, exist_ok=True)
    cgitb.enable(format='text', logdir=log_dir)

    # init GUI
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 高分辨率
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = MainWindow()

    # redirect stdout
    stdout = StdOut()
    stdout.signalForText.connect(mainwindow.displayLog)
    sys.stdout = stdout
    sys.stderr = stdout
    logging.StreamHandler(stdout)

    # show main window
    mainwindow.show()
    sys.exit(app.exec_())
