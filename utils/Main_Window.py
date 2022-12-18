# -*coding=utf-8

import logging
import os
import sys
import time
from datetime import datetime

import cv2
import numpy
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QMessageBox, QFileDialog

from utils import detect, general
from utils.resource import Yolo2onnx_detect_Demo_UI, resource_rc

resource_rc.qInitResources()


class StdOut(QtCore.QObject):
    """rewrite sys.Stdout"""
    signalForText = QtCore.pyqtSignal(str)

    def write(self, text):
        if text == '\n': return
        if not isinstance(text, str): text = str(text)
        if len(text) > 500: text = text[0:10] + ' ...... ' + text[-10:-1]
        self.signalForText.emit(text)

    def flush(self):
        pass


class PyHighlighter(QtGui.QSyntaxHighlighter):
    """格式化python代码"""
    Rules = []
    Formats = {}

    def __init__(self, parent=None):
        super(PyHighlighter, self).__init__(parent)

        self.initializeFormats()

        KEYWORDS = ("and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else", "except", "exec",
                    "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "not", "or", "pass",
                    "print", "raise", "return", "try", "while", "with", "yield")
        BUILTINS = ("abs", "all", "any", "basestring", "bool", "callable", "chr", "classmethod", "cmp", "compile",
                    "complex", "delattr", "dict", "dir", "divmod", "enumerate", "eval", "execfile", "exit", "file",
                    "filter", "float", "frozenset", "getattr", "globals", "hasattr", "hex", "id", "int", "isinstance",
                    "issubclass", "iter", "len", "list", "locals", "map", "max", "min", "object", "oct", "open", "ord",
                    "pow", "property", "range", "reduce", "repr", "reversed", "round", "set", "setattr", "slice",
                    "sorted", "staticmethod", "str", "sum", "super", "tuple", "type", "vars", "zip")
        CONSTANTS = ("False", "True", "None", "NotImplemented", "Ellipsis")

        PyHighlighter.Rules.append(
            (QtCore.QRegExp("|".join([r"\b%s\b" % keyword for keyword in KEYWORDS])), "keyword"))
        PyHighlighter.Rules.append(
            (QtCore.QRegExp("|".join([r"\b%s\b" % builtin for builtin in BUILTINS])), "builtin"))
        PyHighlighter.Rules.append(
            (QtCore.QRegExp("|".join([r"\b%s\b" % constant for constant in CONSTANTS])), "constant"))
        PyHighlighter.Rules.append((QtCore.QRegExp(r"\b[+-]?[0-9]+[lL]?\b"
                                                   r"|\b[+-]?0[xX][0-9A-Fa-f]+[lL]?\b"
                                                   r"|\b[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\b"), "number"))
        PyHighlighter.Rules.append((QtCore.QRegExp(r"\bPyQt4\b|\bQt?[A-Z][a-z]\w+\b"), "pyqt"))
        PyHighlighter.Rules.append((QtCore.QRegExp(r"\b@\w+\b"), "decorator"))
        stringRe = QtCore.QRegExp(r"""(?:'[^']*'|"[^"]*")""")
        stringRe.setMinimal(True)
        PyHighlighter.Rules.append((stringRe, "string"))
        self.stringRe = QtCore.QRegExp(r"""(:?"["]".*"["]"|'''.*''')""")
        self.stringRe.setMinimal(True)
        PyHighlighter.Rules.append((self.stringRe, "string"))
        self.tripleSingleRe = QtCore.QRegExp(r"""'''(?!")""")
        self.tripleDoubleRe = QtCore.QRegExp(r'''"""(?!')''')

    @staticmethod
    def initializeFormats():
        baseFormat = QtGui.QTextCharFormat()
        baseFormat.setFontFamily("courier")
        baseFormat.setFontPointSize(12)
        for name, color in (("normal", QtCore.Qt.black),
                            ("keyword", QtCore.Qt.darkBlue),
                            ("builtin", QtCore.Qt.darkRed),
                            ("constant", QtCore.Qt.darkGreen),
                            ("decorator", QtCore.Qt.darkBlue),
                            ("comment", QtCore.Qt.darkGreen),
                            ("string", QtCore.Qt.darkYellow),
                            ("number", QtCore.Qt.darkMagenta),
                            ("error", QtCore.Qt.darkRed),
                            ("pyqt", QtCore.Qt.darkCyan)):
            format_ = QtGui.QTextCharFormat(baseFormat)
            format_.setForeground(QtGui.QColor(color))
            if name in ("keyword", "decorator"):
                format_.setFontWeight(QtGui.QFont.Bold)
            if name == "comment":
                format_.setFontItalic(True)
            PyHighlighter.Formats[name] = format_

    def highlightBlock(self, text):
        NORMAL, TRIPLESINGLE, TRIPLEDOUBLE, ERROR = range(4)

        textLength = len(text)
        prevState = self.previousBlockState()

        self.setFormat(0, textLength, PyHighlighter.Formats["normal"])

        if text.startswith("Traceback") or text.startswith("Error: "):
            self.setCurrentBlockState(ERROR)
            self.setFormat(0, textLength, PyHighlighter.Formats["error"])
            return
        # noinspection PyTypeChecker
        if prevState == ERROR and not (text.startswith(sys.ps1) or text.startswith("#")):
            self.setCurrentBlockState(ERROR)
            self.setFormat(0, textLength, PyHighlighter.Formats["error"])
            return

        for regex, format_ in PyHighlighter.Rules:
            i = regex.indexIn(text)
            while i >= 0:
                length = regex.matchedLength()
                self.setFormat(i, length, PyHighlighter.Formats[format_])
                i = regex.indexIn(text, i + length)

        # Slow but good quality highlighting for comments. For more
        # speed, comment this out and add the following to __init__:
        # PyHighlighter.Rules.append((QRegExp(r"#.*"), "comment"))
        if not text:
            pass
        elif text[0] == "#":
            self.setFormat(0, len(text), PyHighlighter.Formats["comment"])
        else:
            stack = []
            for i, c in enumerate(text):
                if c in ('"', "'"):
                    if stack and stack[-1] == c:
                        stack.pop()
                    else:
                        stack.append(c)
                elif c == "#" and len(stack) == 0:
                    self.setFormat(i, len(text), PyHighlighter.Formats["comment"])
                    break

        self.setCurrentBlockState(NORMAL)

        if self.stringRe.indexIn(text) != -1:
            return
        # This is fooled by triple quotes inside single quoted strings
        for i, state in ((self.tripleSingleRe.indexIn(text), TRIPLESINGLE),
                         (self.tripleDoubleRe.indexIn(text), TRIPLEDOUBLE)):
            if self.previousBlockState() == state:
                if i == -1:
                    i = len(text)
                    self.setCurrentBlockState(state)
                self.setFormat(0, i + 3, PyHighlighter.Formats["string"])
            elif i > -1:
                self.setCurrentBlockState(state)
                self.setFormat(i, len(text), PyHighlighter.Formats["string"])

    def rehighlight(self):
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        QtGui.QSyntaxHighlighter.rehighlight(self)
        QtWidgets.QApplication.restoreOverrideCursor()


class DetectThread(QtCore.QThread):
    """检测线程"""
    img_sig = QtCore.pyqtSignal(numpy.ndarray)
    res_sig = QtCore.pyqtSignal(dict)

    def __init__(self, model: detect.YOLOv5 = None, dataset: detect.DataLoader = None):
        super(DetectThread, self).__init__()
        self.is_pause = False
        self.is_running = False
        self.is_detecting = False
        self.model = model
        self.dataset: detect.DataLoader = dataset
        self.display_fps = True
        self.print_result = True

    def stopThread(self):
        if not self.is_running:
            return
        self.is_running = False
        self.is_detecting = False

    def stopDetect(self):
        if not self.is_detecting:
            return
        self.is_detecting = False

    def startThread(self):
        if self.is_running:
            return
        self.is_detecting = False
        self.is_running = True
        if not self.isRunning():
            self.start()

    def startDetect(self):
        if self.is_detecting:
            return
        self.is_detecting = True
        self.is_running = True
        if not self.isRunning():
            self.start()

    def pauseDetect(self):
        self.is_pause = True

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
                    res = self.model.detect(img)
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
            if self.display_fps and not self.dataset.is_image:
                fps_ = f'FPS:{fps}'
                lw = max(round(sum(img.shape) / 2 * 0.003), 2)
                tf = max(lw - 1, 1)
                h = cv2.getTextSize(fps_, 0, lw / 3, tf)[0][1]
                cv2.putText(img, fps_, (10, 10 + h), 0, lw / 3, (255, 0, 0), tf, cv2.LINE_AA)

            self.img_sig.emit(img)
            self.res_sig.emit(res)
            # time.sleep(0.0001)

        print(f'"{self.dataset.path}" finished.')
        del self.dataset
        self.stopThread()

    def run(self) -> None:
        try:
            self.main()
        except Exception as e:
            logging.exception(e)


# noinspection PyAttributeOutsideInit
class MainWindow(QtWidgets.QMainWindow, Yolo2onnx_detect_Demo_UI.Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # setup ui, connect callback
        self.setupUi(self)
        self.statusBar().showMessage('initializing...')
        self.UI()
        self.animation()

        # init params
        self.self_script_path = ''
        self.save_video = False
        self.source = ''
        self.video_writer: cv2.VideoWriter
        self.box_color = (255, 0, 0)

        # install event
        self.installEventFilter(self)
        self.textEdit.installEventFilter(self)
        self.textEdit_2.installEventFilter(self)
        self.setAcceptDrops(True)

        # init detect thread
        self.dt = DetectThread()
        self.dt.img_sig.connect(self.displayImg)
        self.dt.res_sig.connect(self.exec)
        self.dt.finished.connect(lambda: self.statusBar().showMessage('exit'))
        self.dt.finished.connect(self.stop)

        # init script api
        self.script_api = ScriptDataAPI(self.dt)
        self.script_api.start_sig.connect(lambda x: self.start() if x else self.stop())

        self.loadConfig()

        self.statusBar().showMessage('initialized', 5000)

    def UI(self):  # 槽函数
        self.toolButton.clicked.connect(self.changeOutputPath)  # 选择保存位置
        self.toolButton_2.clicked.connect(self.changeMediaFile)  # 选择媒体文件
        self.toolButton_3.clicked.connect(self.changeModelFile)  # 选择模型
        self.pushButton_4.clicked.connect(self.start)  # 开始检测槽函数
        self.pushButton_5.clicked.connect(self.stop)
        self.comboBox.currentIndexChanged.connect(self.indexChanged)  # 输入方式切换
        self.pushButton_3.clicked.connect(lambda: self.saveToFile(self.pushButton_3))  # 保存日志
        self.pushButton.clicked.connect(lambda: self.saveToFile(self.pushButton))  # 保存截图
        self.checkBox_4.clicked.connect(lambda: self.saveToFile(self.checkBox_4))  # 保存视频
        self.toolButton_4.clicked.connect(self.changeClassFile)  # 选择类别
        self.textEdit.textChanged.connect(self.displayClassNum)  # 显示类别个数
        self.pushButton_2.clicked.connect(lambda: os.popen(f'explorer "{self.lineEdit.text()}"'))  # 打开保存目录
        self.toolButton_5.clicked.connect(lambda: self.changePyFile(None))  # 导入自定义脚本 槽函数
        self.pushButton_7.clicked.connect(lambda: self.changePyFile(self.self_script_path))  # 选择自定义脚本路径
        self.highlighter = PyHighlighter(self.textEdit_2.document())  # 实例化 格式化代码类
        self.textBrowser.anchorClicked.connect(lambda x: os.popen(f'"{x.toLocalFile()}"'))  # 超链接打开本地文件
        self.checkBox.stateChanged.connect(self.displayFps)  # 帧数显示
        self.checkBox_5.clicked.connect(self.printResult)  # 打印检测结果
        self.checkBox_6.clicked.connect(self.changeModelConfig)  # 是否返回坐标
        self.checkBox_2.clicked.connect(self.changeModelConfig)  # 是否画锚框
        self.doubleSpinBox.valueChanged.connect(self.changeModelConfig)  # 更改置信度
        self.doubleSpinBox_2.valueChanged.connect(self.changeModelConfig)  # 更改IOU
        self.toolButton_6.clicked.connect(self.changeBoxColor)  # 更改锚框颜色
        self.checkBox_3.clicked.connect(lambda x: print(f'script {x}'))
        self.pushButton_8.toggled.connect(self.lockBottom)  # 锁定切换 槽函数
        self.pushButton_10.clicked.connect(lambda: self.textBrowser.clear())  # 清空控制台
        self.pushButton_9.clicked.connect(self.resetSource)  # 重置输入源

        self.pushButton_6.setHidden(True)  # 暂停按钮, 暂时隐藏

    def animation(self):
        self.animation_1 = QtCore.QPropertyAnimation(self.textEdit, b"maximumHeight", self)
        self.animation_1.setStartValue(200)
        self.animation_1.setEndValue(20)
        self.animation_1.setDuration(100)

    def loadConfig(self):  # 加载配置
        cfg = general.cfg('config.cfg')
        # input_source
        self.setSource(cfg.search('root', 'input_source', default_value=0))
        # model_path
        self.lineEdit_3.setText(cfg.search('root', 'model_path', os.path.join('need', 'models', 'yolov7-tiny.onnx')))
        # class_path
        self.class_file = cfg.search('root', 'class_path', os.path.join('need', 'yolov7-tiny.txt'))
        if os.path.exists(self.class_file):
            self.textEdit.setText(open(self.class_file, 'r').read().replace('\n', ','))  # 初始化类别
        # display_fps
        self.checkBox.setChecked(cfg.search('root', 'display_fps', default_value=True, return_type=bool))
        # conf_thres
        self.doubleSpinBox.setValue(cfg.search('root', 'conf_thres', default_value=0.5, return_type=float))
        # iou_thres
        self.doubleSpinBox_2.setValue(cfg.search('root', 'iou_thres', default_value=0.5, return_type=float))
        # display_box
        self.checkBox_2.setChecked(cfg.search('root', 'display_box', default_value=True, return_type=bool))
        # box_color
        self.changeBoxColor(eval(cfg.search('root', 'box_color', default_value='(255,0,0)')))
        # print_result
        self.checkBox_5.setChecked(cfg.search('root', 'print_result', default_value=True, return_type=bool))
        # with_pos
        self.checkBox_6.setChecked(cfg.search('root', 'with_pos', default_value=False, return_type=bool))
        # record_video
        self.checkBox_4.setChecked(cfg.search('root', 'record_video', default_value=False, return_type=bool))
        # record_fps
        self.spinBox.setValue(cfg.search('root', 'record_fps', default_value=15, return_type=int))
        # out_path
        self.lineEdit.setText(cfg.search('root', 'out_path', default_value=os.path.join(os.getcwd(), 'out')))
        # script_path
        self.changePyFile(cfg.search('root', 'script_path', default_value=os.path.join('need', 'self_demo.py')))
        # script_status
        self.checkBox_3.setChecked(cfg.search('root', 'script_status', default_value=False, return_type=bool))
        # detect_status
        if cfg.search('root', 'detect_status', default_value=False, return_type=bool):
            self.start()

    def saveConfig(self):  # 保存配置
        with general.cfg('config.cfg') as cfg:
            cfg.set('root', 'detect_status', self.dt.is_detecting)
            cfg.set('root', 'input_source', self.source)
            cfg.set('root', 'model_path', self.lineEdit_3.text())
            cfg.set('root', 'class_path', self.class_file)
            cfg.set('root', 'display_fps', self.checkBox.isChecked())
            cfg.set('root', 'conf_thres', self.doubleSpinBox.value())
            cfg.set('root', 'iou_thres', self.doubleSpinBox_2.value())
            cfg.set('root', 'display_box', self.checkBox_2.isChecked())
            cfg.set('root', 'box_color', self.box_color)
            cfg.set('root', 'print_result', self.checkBox_5.isChecked())
            cfg.set('root', 'with_pos', self.checkBox_6.isChecked())
            cfg.set('root', 'record_video', self.checkBox_4.isChecked())
            cfg.set('root', 'record_fps', self.spinBox.value())
            cfg.set('root', 'out_path', self.lineEdit.text())
            cfg.set('root', 'script_path', self.self_script_path)
            cfg.set('root', 'script_status', self.checkBox_3.isChecked())

    def changeModelConfig(self):  # 更改模型配置
        if self.dt.model is None:
            return
        self.dt.model.conf_threshold = self.doubleSpinBox.value()  # 更改置信度
        self.dt.model.iou_threshold = self.doubleSpinBox_2.value()  # 更改IOU
        self.dt.model.draw_box = self.checkBox_2.isChecked()  # 是否显示锚框
        self.dt.model.with_pos = self.checkBox_6.isChecked()  # 是否返回坐标

    def changeBoxColor(self, color: tuple = None):  # 更改锚框颜色
        if not color:
            old_color = self.dt.model.box_color if self.dt.model else self.box_color
            new_color = QtWidgets.QColorDialog.getColor(QtGui.QColor(*old_color))
            if not new_color.isValid():
                return
            color = new_color.getRgb()[0:3]
        self.toolButton_6.setStyleSheet(f"color:rgb{color}")
        self.box_color = color
        if self.dt.model is None:
            return
        self.dt.model.box_color = self.box_color
        self.dt.model.txt_color = tuple(255 - x for x in self.box_color)

    def printResult(self):  # 打印检测结果
        self.dt.print_result = self.checkBox_5.isChecked()

    def saveToFile(self, index):  # 保存截图、视频、日志
        os.makedirs(self.lineEdit.text(), exist_ok=True)
        head = datetime.now().strftime('%m-%d %H-%M-%S')
        # 保存截图
        if index == self.pushButton:
            path = os.path.join(self.lineEdit.text(), f'ScreenShot_{head}.png')
            if self.label.pixmap() is None:
                return
            self.label.pixmap().toImage().save(path)
            print(f'ScreenShot has been saved to <a href="file:///{path}">{path}</a>')
        # 保存日志
        elif index == self.pushButton_3:
            path = os.path.join(self.lineEdit.text(), f'log_{head}.log')
            with open(path, 'w') as f:
                f.write(self.textBrowser.toPlainText())
                print(f'Log has been saved to <a href="file:///{path}">{path}</a>')
        # 保存录屏视频
        elif index == self.checkBox_4 and self.checkBox_4.isChecked() and self.dt.is_detecting:
            if self.dt.dataset.is_image:
                return
            self.save_video_path = os.path.join(self.lineEdit.text(), f'video_{head}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编解码器
            fps = self.spinBox.value()
            width, height = self.dt.dataset.w, self.dt.dataset.h  # 宽高
            self.video_writer = cv2.VideoWriter(self.save_video_path, fourcc, fps, (width, height))  # 写入视频
            self.save_video = True
            print(f'begin recording...')
        elif index == self.checkBox_4 and not self.checkBox_4.isChecked():
            self.stopRecord()

    def stopRecord(self):  # 停止录制
        if not self.save_video:
            return
        self.save_video = False
        self.video_writer.release()
        print(f'Video has been saved to <a href="file:///{self.save_video_path}">{self.save_video_path}</a>')

    def resetSource(self):  # 重置输入源
        if self.dt.is_detecting:
            return
        self.indexChanged(self.comboBox.currentIndex())

    def indexChanged(self, index):  # 切换输入方式
        self.dt.blockSignals(True)
        self.dt.stopThread()
        self.dt.wait()
        if index == 0:  # webcam
            self.setSource('0')
        elif index == 1 and os.path.exists(self.lineEdit_2.text()):  # file
            self.setSource(self.lineEdit_2.text(), frame_skip=-1)
        elif index == 2 or index == 4:  # url or custom data
            text, flag = QtWidgets.QInputDialog.getText(self, 'Custom Source', 'input:', text=self.source)
            if not flag:
                return
            self.setSource(text)
        elif index == 3:  # screen
            self.setSource('screen', frame_skip=-1)
        self.dt.blockSignals(False)

    def setSource(self, source, **kwargs) -> bool:  # 设置输入源
        self.source = str(source)
        try:
            self.dt.dataset = detect.DataLoader(self.source, **kwargs)
        except Exception as e:
            self.label.setText(str(e))
            self.displayLog(str(e), color='red')
            return False
        self.stopRecord()
        self.comboBox.blockSignals(True)
        index = (self.source == '0',
                 self.dt.dataset.is_image or self.dt.dataset.is_video,
                 self.dt.dataset.is_url,
                 self.source.lower() == 'screen',
                 True).index(True)
        self.comboBox.setCurrentIndex(index)
        self.comboBox.blockSignals(False)
        if self.dt.dataset.is_wabcam or self.dt.dataset.is_url or self.dt.dataset.is_screen:
            self.dt.startThread()
        elif self.dt.dataset.is_image or self.dt.dataset.is_video:
            self.lineEdit_2.setText(self.source)
            vc = cv2.VideoCapture(self.source)
            _, img = vc.read()
            self.displayImg(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            vc.release()
        return 'dataset' in self.dt.__dict__.keys()

    def start(self):  # 启动检测线程
        if self.dt.is_detecting:
            print('already running')
            return
        if not os.path.exists(self.lineEdit_3.text()):
            self.displayLog(f'"{self.lineEdit_3.text()}" not exist', color='red')
            return
        if self.comboBox.currentIndex() == 1 and not os.path.exists(self.lineEdit_2.text()):  # 视频
            self.displayLog(f'"{self.lineEdit_2.text()}" not exist', color='red')
            return
        if 'dataset' not in self.dt.__dict__.keys() and not self.setSource(self.source):
            return
        self.dt.model = detect.YOLOv5()
        self.dt.model.initConfig(input_width=640,
                                 input_height=640,
                                 conf_thres=self.doubleSpinBox.value(),
                                 iou_thres=self.doubleSpinBox_2.value(),
                                 draw_box=self.checkBox_2.isChecked(),
                                 thickness=2,
                                 class_names=self.textEdit.toPlainText().split(','),
                                 box_color=self.box_color,
                                 txt_color=tuple(255 - x for x in self.box_color),
                                 with_pos=self.checkBox_6.isChecked(),
                                 )

        self.dt.model.initModel(self.lineEdit_3.text(), t='onnxruntime')  # cv2.dnn or onnxruntime

        self.dt.startDetect()
        self.saveToFile(self.checkBox_4)
        print('start detect')
        self.statusBar().showMessage('start detect...', 5000)

    # todo
    def pause(self):  # 暂停
        pass

    def stop(self):  # 停止检测
        self.stopRecord()
        if self.dt.is_detecting:
            self.dt.stopDetect()
            self.statusBar().showMessage('stop detect', 5000)
            print('stop detect')
            return
        if self.dt.is_running:
            self.dt.stopThread()
            self.dt.wait()

    def changeModelFile(self):  # 选择权重文件
        path, _ = QFileDialog.getOpenFileName(self, "选择模型",
                                              os.path.abspath(self.lineEdit_3.text()),
                                              '*.onnx')
        if path:
            self.lineEdit_3.setText(path)
        class_txt_path = os.path.join(os.path.dirname(os.path.dirname(path)),
                                      ''.join(os.path.basename(path).split('.')[:-1]) + '.txt')
        if os.path.exists(class_txt_path):
            self.class_file = class_txt_path
            with open(self.class_file, 'r') as f:
                self.textEdit.setText(f.read().replace('，', ',').replace('|', ',').replace('\n', ','))

    def changeClassFile(self):  # 选择类别文件
        path, _ = QFileDialog.getOpenFileName(self, "选择文件",
                                              os.path.abspath(self.class_file),
                                              '*.txt')
        if path:
            self.class_file = path
            with open(self.class_file, 'r') as f:
                self.textEdit.setText(f.read().replace('，', ',').replace('|', ',').replace('\n', ','))

    def changeMediaFile(self):  # 选择媒体文件
        path, _ = QFileDialog.getOpenFileName(self, "选择文件",
                                              os.path.abspath(self.lineEdit_2.text()),
                                              '*.asf *.avi *.gif *.m4v *.mkv *.mov *.mp4 *.mpeg *.mpg *.ts *.wmv '
                                              '*.bmp *.dng *.jpeg *.jpg *.mpo *.png *.tif *.tiff *.webp *.pfm')
        if path:
            self.setSource(path)

    def changeOutputPath(self):  # 选择保存位置
        file_path = QFileDialog.getExistingDirectory(self, "选择保存位置", self.lineEdit.text())
        if file_path:
            self.lineEdit.setText(file_path)

    def changePyFile(self, path=None):  # 选择自定义脚本
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件",
                                                  os.path.abspath(self.self_script_path),
                                                  'Python File(*.py *.pyw)')
            if not path:
                return
        self.self_script_path = path
        if not os.path.exists(self.self_script_path):
            self.displayLog(f'"{self.self_script_path}" not exist', color='red')
            return
        for i in ('utf8', 'gbk'):
            try:
                self.textEdit_2.setPlainText(open(self.self_script_path, 'r', encoding=i).read())
                sys.path.append(os.path.dirname(os.path.abspath(self.self_script_path)))
            except UnicodeError:
                pass

    def displayClassNum(self):  # 显示类别数量
        class_set = set(self.textEdit.toPlainText().split(","))
        class_set.discard('')
        self.label_6.setText(f'类别({len(class_set)}):')

    def displayFps(self):  # 显示FPS
        self.dt.display_fps = self.checkBox.isChecked()

    def displayImg(self, img: numpy.ndarray):  # 显示图片到label
        if self.save_video:
            self.video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.script_api.img_data = img
        img = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QtGui.QImage.Format_RGB888)
        p = min(self.label.width() / img.width(), self.label.height() / img.height())
        pix = QtGui.QPixmap(img).scaled(int(img.width() * p), int(img.height() * p))
        self.label.setPixmap(pix)

    def displayLog(self, text: str, color='black', plain_text=False):  # 输出控制台信息到textBrowser
        head_ = f"{datetime.now().strftime('%H:%M:%S.%f')} >> "

        if text.startswith(('<',)) or plain_text:
            self.textBrowser.setTextColor(QtGui.QColor('black'))
            self.textBrowser.append(head_)
            self.textBrowser.setTextColor(QtGui.QColor(color))
            self.textBrowser.insertPlainText(text)
        else:
            text = f"{head_}<font color='{color}'>{text}"
            self.textBrowser.append(text)
        # 自动切换锁定状态
        scrollbar = self.textBrowser.verticalScrollBar()
        self.pushButton_8.setChecked(scrollbar.value() >= scrollbar.maximum())
        if self.pushButton_8.isChecked():
            scrollbar.setValue(scrollbar.maximum())

    def lockBottom(self, status):  # 锁定底部切换
        scrollbar = self.textBrowser.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum() if status else scrollbar.maximum() - 1)

    # todo: 更优雅
    def exec(self, text: dict):  # 执行自定义脚本
        if not self.checkBox_3.isChecked():
            return
        try:
            def scriptPrint(*args, **kwargs):
                head = '[' + os.path.basename(self.self_script_path) + ']: '
                for i in args:
                    head = head + str(i) + ' '
                self.displayLog(head, color='gray', plain_text=True)

            globals_ = globals().copy()
            globals_['data_api'] = self.script_api
            globals_['print'] = scriptPrint
            self.script_api.res_data = text
            exec(self.textEdit_2.toPlainText(), globals_)
        except Exception as e:
            self.checkBox_3.setChecked(False)
            logging.exception(e)

    def eventFilter(self, objwatched, event):  # 重写事件过滤
        eventType = event.type()
        # 替换tab为4个空格
        if objwatched == self.textEdit_2 and eventType == QtCore.QEvent.KeyPress and event.key() == QtCore.Qt.Key_Tab:
            cursor = self.textEdit_2.textCursor()
            cursor.insertText(" " * 4)
            return True
        # 设置/取消焦点
        if eventType == QtCore.QEvent.MouseButtonPress:
            self.setFocus()
        # 快捷键: ctrl+r开始; ctrl+e停止; ctrl+s截图
        if eventType == QtCore.QEvent.KeyPress:
            if event.modifiers() == QtCore.Qt.ControlModifier and event.key() == QtCore.Qt.Key_R:
                self.start()
            elif event.modifiers() == QtCore.Qt.ControlModifier and event.key() == QtCore.Qt.Key_E:
                self.stop()
            elif event.modifiers() == QtCore.Qt.ControlModifier and event.key() == QtCore.Qt.Key_S:
                self.saveToFile(self.pushButton)
        # 捕获拖入文件
        if eventType == QtCore.QEvent.DragEnter:
            event.accept()
        # 判断并设置拖入文件路径
        if eventType == QtCore.QEvent.Drop:
            for file in event.mimeData().urls():
                f: str = file.toLocalFile()
                if not os.path.isfile(f):
                    return False
                if f.lower().endswith('.onnx'):
                    self.lineEdit_3.setText(f)
                elif f.lower().endswith(detect.DataLoader.IMAGE_TYPE + detect.DataLoader.VIDEO_TYPE):
                    self.lineEdit_2.setText(f)
                elif f.lower().endswith('.txt'):
                    self.class_file = f
                    with open(self.class_file, 'r') as f_:
                        self.textEdit.setText(f_.read().replace('，', ',').replace('|', ',').replace('\n', ','))
                elif f.lower().endswith(('.py', '.pyw')):
                    self.changePyFile(f)
        # 关闭窗口事件
        if eventType == QtCore.QEvent.Close:
            if self.dt.is_detecting:
                msgbox = QMessageBox.question(self,
                                              self.windowTitle(),
                                              '正在运行\n是否停止并关闭?',
                                              QMessageBox.Yes | QMessageBox.Ignore | QMessageBox.No,
                                              QMessageBox.Yes)
                if msgbox == QMessageBox.Yes:
                    self.stop()
                elif msgbox == QMessageBox.No:
                    event.ignore()
                    return
            self.saveConfig()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        # 折叠class_list
        if objwatched == self.textEdit:
            if eventType == QtCore.QEvent.FocusIn:
                self.animation_1.setDirection(QtCore.QAbstractAnimation.Backward)
                self.animation_1.start()
            elif eventType == QtCore.QEvent.FocusOut:
                self.animation_1.setDirection(QtCore.QAbstractAnimation.Forward)
                self.animation_1.start()
        return super().eventFilter(objwatched, event)


class ScriptDataAPI(QtCore.QObject):
    """
    res_data: 属性, 检测结果, dict类型
    img_data: 属性, 实时图片, numpy.ndarray类型
    setDetectStatus(bool): 方法, 设置检测状态, 接受一个bool类型参数, 为True时开启检测, 为False时停止检测
    setModelConfig(**kwargs): 方法, 设置模型配置, 接受关键字参数, 可选关键字参数有
        model_path(str),
        input_width(int),
        input_height(int),
        draw_box(bool),
        box_color(tuple[int,int,int]),
        txt_color(tuple[int,int,int]),
        thickness(int),
        conf_thres(float),
        iou_thres(float),
        class_names(list[str])
    """
    start_sig = QtCore.pyqtSignal(bool)

    def __init__(self, thread):
        super(ScriptDataAPI, self).__init__()
        self.thread: DetectThread = thread
        self.res_data: dict = {}
        self.img_data: numpy.ndarray

    @property
    def help(self):
        return self.__doc__

    def setDetectStatus(self, status: bool):
        if status and not self.thread.is_detecting:
            self.start_sig.emit(status)
        elif not status and self.thread.is_detecting:
            self.start_sig.emit(status)

    def setModelConfig(self, **kwargs):
        if 'model_path' in kwargs:
            self.thread.model.initModel(kwargs['model_path'])
        if self.thread.model:
            self.thread.model.__dict__.update(**kwargs)
