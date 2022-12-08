# -*coding=utf-8

import argparse
import cgitb
import logging
import os
import sys
import time

import cv2


def run(**kwargs):
    if not kwargs['nogui']:  # GUI
        from PyQt5 import QtWidgets, QtCore
        from utils.Main_Window import MainWindow, StdOut

        # init GUI
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 高分辨率
        app = QtWidgets.QApplication(sys.argv)
        mainwindow = MainWindow()
        # import qt_material  # 一个美化pyqt界面的第三方库
        # qt_material.apply_stylesheet(app, 'dark_blue.xml')

        # redirect stdout
        stdout = StdOut()
        stdout.signalForText.connect(mainwindow.displayLog)
        sys.stdout = stdout
        sys.stderr = stdout
        logging.StreamHandler(stdout)

        # show main window
        mainwindow.show()
        sys.exit(app.exec_())
    else:  # 命令行
        from utils import detect

        if os.path.isdir(kwargs['source']):
            file = [os.path.join(kwargs['source'], x) for x in os.listdir(kwargs['source'])]
        elif os.path.isfile(kwargs['source']):
            file = (kwargs['source'],)
        else:
            raise ValueError
        model = detect.YOLOv5()
        model.initModel(kwargs['weights'])
        if not os.path.exists(kwargs['classes']):
            raise FileNotFoundError
        class_names = open(kwargs['classes'], 'r').read().split('\n')
        model.initConfig(input_width=kwargs['imgsz'][0],
                         input_height=kwargs['imgsz'][1],
                         conf_thres=kwargs['conf_thres'],
                         iou_thres=kwargs['iou_thres'],
                         draw_box=True,
                         thickness=2,
                         class_names=class_names,
                         box_color=(0, 0, 255),
                         txt_color=(255, 255, 0),
                         with_pos=False,
                         )
        print('\nstart detect, press "ctrl+c" to quit\n')
        for f in file:
            try:
                dataset = detect.DataLoader(f, 0)
            except AssertionError:
                continue

            start_time = time.time()
            with open(os.path.join(kwargs['save_path'], os.path.basename(f)+'.log'), 'w'): pass
            log_file = open(os.path.join(kwargs['save_path'], os.path.basename(f)+'.log'), 'a')
            if dataset.is_video and (not kwargs['video_split']):
                dst_path = os.path.join(kwargs['save_path'], os.path.basename(f))
                print(dst_path)
                out_v = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*'mp4v'), dataset.fps, (dataset.w, dataset.h))

            count = 0
            for img, base_name in dataset:
                res_ = model.detect(img)
                count += 1
                log_file.write(f'{count}: {res_}\n')
                if dataset.is_image:
                    dst_path = os.path.join(kwargs['save_path'], base_name)
                    cv2.imwrite(dst_path, img)
                    print(dst_path)
                elif dataset.is_video:
                    if kwargs['video_split']:
                        dst_path = os.path.join(kwargs['save_path'], base_name)
                        os.makedirs(dst_path, exist_ok=True)
                        cv2.imwrite(os.path.join(dst_path, f'{base_name}_{count}.png'), img)
                        # print(os.path.join(dst_path, base_name))
                    else:
                        out_v.write(img)
            log_file.close()
            print('finished in %.3f s' % (time.time() - start_time))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nogui', action='store_true')
    parser.add_argument('--weights', type=str, default='need/models/yolov7-tiny.onnx')
    parser.add_argument('--classes', type=str, default='need/yolov7-tiny.txt')
    parser.add_argument('--source', type=str, default='data')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='inference size w,h')
    parser.add_argument('--conf_thres', type=float, default=0.5)
    parser.add_argument('--iou_thres', type=float, default=0.5)
    parser.add_argument('--save_path', type=str, default='out')
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--save_result', action='store_true')
    parser.add_argument('--video_split', action='store_true')
    opt_ = parser.parse_args()
    opt_.imgsz *= 2 if len(opt_.imgsz) == 1 else 1  # expand
    print(vars(opt_))
    return opt_


if __name__ == "__main__":
    # dump logs
    log_dir = os.path.join(os.getcwd(), 'log')
    os.makedirs(log_dir, exist_ok=True)
    cgitb.enable(format='text', logdir=log_dir)

    opt = parse_opt()
    run(**vars(opt))
