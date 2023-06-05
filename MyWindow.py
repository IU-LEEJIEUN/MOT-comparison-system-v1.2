import os
import shutil
import sys

from PyQt5 import uic, QtGui, QtCore, QtWidgets
from PyQt5.QtCore import QDateTime, QTimer, QTime, Qt, QThread, QEventLoop
from PyQt5.QtGui import QIcon, QPixmap, QImage, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QMessageBox, QTableWidgetItem

from Move2Center import center
from CircularQueue import CircularQueue
import cv2
from tracker.mutil_track import mutil_track

import resources_rc


def convert_to_pixmap(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(image)
    return pixmap


def read_summary_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        keys = lines[0].split()
        values = lines[1].split()
        return dict(zip(keys, values))


def get_summary_dict(path, prefix):
    result = {}
    for root, dirs, files in os.walk(path):
        for d in dirs:
            if d.startswith(prefix):
                dir_path = os.path.join(root, d)
                d = d.split('_')[0]
                for file in os.listdir(dir_path):
                    if file.endswith('summary.txt'):
                        file_path = os.path.join(dir_path, file)
                        result[d] = read_summary_file(file_path)
    return result


class Track_Thread(QThread):
    def __init__(self, track1, track2, track3, video_path, filename):
        super().__init__()
        self.track1 = track1
        self.track2 = track2
        self.track3 = track3
        self.video_path = video_path
        self.filename = filename

    def run(self):
        mutil_track(self.track1, self.track2, self.track3, self.video_path, self.filename)


def clear_cache():
    path = './multi_track_result'
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        pass


class MyWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.ui = None
        self.init_ui()
        self.tracked_result1 = None
        self.tracked_result2 = None
        self.tracked_result3 = None
        self.algo_list = []
        self.total_frames = 0  # total frames
        self.mFrameRate = 0  # frame rate
        self.mIsBroadcast = False  # broadcast flag
        self.mCurrentFrame = 0  # current frame
        self.mFrame = None  # picture frame information
        self.input_frame = 0  # the frame you want to skip
        self.msg_box = QMessageBox()  # various Message Box
        self.ALGOName_queue = CircularQueue(3)
        self.msg_box.setWindowIcon(QIcon("./pic/logo.png"))
        self.ui.SkipBtn.setShortcut(QtCore.Qt.Key_Return)  # Set the shortcut key to trigger skip
        self.ui.FrameEdit.setValidator(QtGui.QIntValidator())  # Only numbers can be inputted in the FrameEdit
        self.mCurrentTime = QDateTime.fromString("00:00:00", "hh:mm:ss")
        self.ui.timeLabel.setText(self.mCurrentTime.toString("hh:mm:ss"))
        self.ui.FrameEdit.setPlaceholderText("number")
        self.ui.actionOpen.triggered.connect(self.process_video)
        self.ui.actionSORT.triggered.connect(self.Select_SORT)
        self.ui.actionDeepSORT.triggered.connect(self.Select_DeepSORT)
        self.ui.actionByteTrack.triggered.connect(self.Select_ByteTrack)
        self.ui.actionC_BIoUTrack.triggered.connect(self.Select_C_BIoUTrack)
        self.ui.actionUAVMOT.triggered.connect(self.Select_UAVMOT)
        self.ui.actionBotSORT.triggered.connect(self.Select_BotSORT)
        self.ui.actionClear.triggered.connect(self.Clear)
        self.ui.horizontalSlider.sliderMoved.connect(self.sliderMoved)
        self.ui.broadcastBtn.clicked.connect(self.broadcast)
        self.ui.stopBtn.clicked.connect(self.stop)
        self.ui.SkipBtn.clicked.connect(self.skip)
        self.ui.PreFrameBtn.clicked.connect(self.pre_frame)
        self.ui.NextFrameBtn.clicked.connect(self.next_frame)
        self.theTimer = QTimer(self)
        self.theTimer.timeout.connect(self.updateImage)
        self.ui.broadcastBtn.setEnabled(False)  # The broadcastBtn cannot be used before opening the video
        self.set_table_style()
        self.setStop_BtnStyle()
        self.setCentral_widgetStyle()
        self.setStatus_barStyle()

    def init_ui(self):
        self.ui = uic.loadUi("./MainWindow.ui")
        self.ui.setWindowIcon(QIcon("./pic/logo.png"))
        self.setBroadcastState_BtnStyle()
        self.setPauseState_BtnStyle()

    def process_video(self):
        clear_cache()
        if not self.ui.ALGOname1.text() or not self.ui.ALGOname2.text() or not self.ui.ALGOname3.text():
            self.msg_box.setIcon(QMessageBox.Warning)
            self.msg_box.setText("Please select three algorithms first")
            self.msg_box.setWindowTitle("Information")
            self.msg_box.exec_()
        else:
            video_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "",
                                                        "Videos (*.mp4 *.avi *.mkv);;All Files (*)")
            if video_path:
                # Read the video and extract frames
                cap = cv2.VideoCapture(video_path)
                # get video properties
                self.mFrameRate = cap.get(cv2.CAP_PROP_FPS)
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                filename = os.path.basename(video_path)
                filename, ext = os.path.splitext(filename)
                assert os.path.exists(video_path), 'the video does not exist! '
                track1 = self.ui.ALGOname1.text()
                track2 = self.ui.ALGOname2.text()
                track3 = self.ui.ALGOname3.text()
                thread = Track_Thread(track1, track2, track3, video_path, filename)
                thread.finished.connect(self.tracked_completed)
                thread.start()
                msg_box = QMessageBox()
                msg_box.setWindowIcon(QIcon("./pic/logo.png"))
                msg_box.setWindowTitle("Tracking")
                msg_box.setText("Please wait for a while")
                msg_box.setStandardButtons(QMessageBox.NoButton)
                msg_box.setWindowFlags(msg_box.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
                msg_box.show()
                tracked_result_folder = './multi_track_result/result_images'
                while not os.path.exists(tracked_result_folder) or len(os.listdir(tracked_result_folder)) != 3:
                    loop = QEventLoop()
                    timer = QTimer()
                    timer.setSingleShot(True)
                    timer.timeout.connect(loop.quit)
                    timer.start(1000)
                    loop.exec_()
                for folder in os.listdir(tracked_result_folder):
                    if os.path.isdir(os.path.join(tracked_result_folder, folder)):
                        if folder.startswith(track1):
                            self.tracked_result1 = os.path.join(tracked_result_folder, folder)
                        elif folder.startswith(track2):
                            self.tracked_result2 = os.path.join(tracked_result_folder, folder)
                        elif folder.startswith(track3):
                            self.tracked_result3 = os.path.join(tracked_result_folder, folder)
                self.ui.broadcastBtn.setEnabled(True)
                self.updateImage()
            else:
                pass

    def tracked_completed(self):
        path = './multi_track_result'
        prefixes = [self.ui.ALGOname1.text(), self.ui.ALGOname2.text(), self.ui.ALGOname3.text()]
        dicts = {}
        for prefix in prefixes:
            dicts[prefix] = get_summary_dict(path, prefix)
        self.read_indicator(dicts, max_cols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14], min_cols=[9, 11])

    def updateImage(self):
        if self.mCurrentFrame >= self.total_frames:
            self.mCurrentFrame = 0
            self.ui.horizontalSlider.setValue(self.mCurrentFrame)
            self.mCurrentTime = QDateTime.fromString("00:00:00", "hh:mm:ss")
            self.ui.timeLabel.setText(self.mCurrentTime.toString("hh:mm:ss"))
            self.theTimer.stop()
            self.ui.VideoWindow1.clear()
            self.ui.VideoWindow2.clear()
            self.ui.VideoWindow3.clear()
            return
        max_frame = 0
        for video_window, track_result in zip([self.ui.VideoWindow1, self.ui.VideoWindow2, self.ui.VideoWindow3],
                                              [self.tracked_result1, self.tracked_result2, self.tracked_result3]):
            if os.path.exists(track_result):
                image_filenames = os.listdir(track_result)
                if image_filenames:
                    image_filenames = [f for f in image_filenames if f.endswith('.jpg')]
                    max_frame = max(max_frame, int(os.path.splitext(image_filenames[-1])[0]) - 1)
            if self.mCurrentFrame > max_frame:
                self.mCurrentFrame = max_frame
            image_filename = os.path.join(track_result, '{:05d}.jpg'.format(self.mCurrentFrame))
            self.mFrame = cv2.imread(image_filename)
            if self.mFrame is not None:
                pixmap = convert_to_pixmap(self.mFrame)
                video_window.setPixmap(pixmap)
                video_window.setScaledContents(True)
                self.ui.CurrentFrame.setText("Current Frame:{}".format(int(self.mCurrentFrame)))
        self.ui.horizontalSlider.setValue(int(self.mCurrentFrame))
        self.mCurrentFrame += 1
        self.ui.horizontalSlider.setMaximum(self.total_frames - 2)
        self.mCurrentTime = QDateTime.fromString("00:00:00", "hh:mm:ss").addMSecs(
            int(1000 * self.mCurrentFrame / self.mFrameRate))
        self.ui.timeLabel.setText(self.mCurrentTime.toString("hh:mm:ss"))
        self.ui.FPS.setText("FPS:{}".format(int(self.mFrameRate)))

    # Slot function of slide moved
    def sliderMoved(self, position):
        if self.ui.VideoWindow1.pixmap() is not None:
            # Update current frame and time for each video window
            max_frame = 0
            for video_window, track_result in zip([self.ui.VideoWindow1, self.ui.VideoWindow2, self.ui.VideoWindow3],
                                                  [self.tracked_result1, self.tracked_result2, self.tracked_result3]):
                if os.path.exists(track_result):
                    image_filenames = os.listdir(track_result)
                    if image_filenames:
                        image_filenames = [f for f in image_filenames if f.endswith('.jpg')]
                        max_frame = max(max_frame, int(os.path.splitext(image_filenames[-1])[0]) - 1)
                if position > max_frame:
                    position = max_frame
                    self.ui.horizontalSlider.setValue(position)
                self.mCurrentFrame = position
                self.mCurrentTime = QDateTime.fromString("00:00:00", "hh:mm:ss").addMSecs(
                    int(1000 * position / self.mFrameRate))
                image_filename = os.path.join(track_result, '{:05d}.jpg'.format(self.mCurrentFrame))
                self.mFrame = cv2.imread(image_filename)
                if self.mFrame is not None:
                    pixmap = convert_to_pixmap(self.mFrame)
                    video_window.setPixmap(pixmap)
                    video_window.setScaledContents(True)
            self.ui.timeLabel.setText(self.mCurrentTime.toString("hh:mm:ss"))
            self.ui.CurrentFrame.setText("Current Frame:{}".format(int(self.mCurrentFrame)))
            self.ui.FPS.setText("FPS:{}".format(int(self.mFrameRate)))
        else:
            self.ui.horizontalSlider.setValue(0)
            self.msg_box.setIcon(QMessageBox.Warning)
            self.msg_box.setText("Please open the video file first")
            self.msg_box.setWindowTitle("Warning")
            self.msg_box.show()

    def broadcast(self):
        if not self.mIsBroadcast:
            self.setPauseState_BtnStyle()
            self.mIsBroadcast = True
            # self.theTimer.start(int(10000 / self.mFrameRate))
            self.theTimer.start(200)
            self.ui.FPS.setText("FPS:{}".format(int(self.mFrameRate)))
        else:
            self.setBroadcastState_BtnStyle()
            self.mIsBroadcast = False
            self.theTimer.stop()

    def stop(self):
        self.setBroadcastState_BtnStyle()
        self.mIsBroadcast = False
        self.mCurrentFrame = 0
        self.ui.horizontalSlider.setValue(self.mCurrentFrame)
        self.mCurrentTime = QDateTime.fromString("00:00:00", "hh:mm:ss")
        self.ui.timeLabel.setText(self.mCurrentTime.toString("hh:mm:ss"))
        self.ui.FPS.setText("FPS:")
        self.ui.CurrentFrame.setText("Current Frame:")
        self.ui.VideoWindow1.clear()
        self.ui.VideoWindow2.clear()
        self.ui.VideoWindow3.clear()
        self.theTimer.stop()

    def skip(self):
        if self.ui.FrameEdit.text() == '':
            self.msg_box.setIcon(QMessageBox.Question)
            self.msg_box.setText("Please input a number")
            self.msg_box.setWindowTitle("Warning")
            self.msg_box.exec_()
        else:
            self.input_frame = int(self.ui.FrameEdit.text())
        if self.input_frame < 0 and self.ui.VideoWindow1.pixmap() is not None:
            self.msg_box.setIcon(QMessageBox.Warning)
            self.msg_box.setText("Please input the correct number of frames")
            self.msg_box.setWindowTitle("Warning")
            self.msg_box.exec_()
        elif self.input_frame > self.total_frames and self.ui.VideoWindow1.pixmap() is not None:
            self.msg_box.setIcon(QMessageBox.Warning)
            self.msg_box.setText("The number you inputted is greater than the total frames of the video."
                                 "Please input the correct number of frames")
            self.msg_box.setWindowTitle("Warning")
            self.msg_box.exec_()
        if self.ui.VideoWindow1.pixmap() is None:
            self.msg_box.setIcon(QMessageBox.Warning)
            self.msg_box.setText("Please open the video file first")
            self.msg_box.setWindowTitle("Warning")
            self.msg_box.exec_()
        elif 0 <= self.input_frame <= self.total_frames:
            max_frame = 0
            self.mCurrentFrame = self.input_frame
            for video_window, track_result in zip([self.ui.VideoWindow1, self.ui.VideoWindow2, self.ui.VideoWindow3],
                                                  [self.tracked_result1, self.tracked_result2, self.tracked_result3]):
                if os.path.exists(track_result):
                    image_filenames = os.listdir(track_result)
                    if image_filenames:
                        image_filenames = [f for f in image_filenames if f.endswith('.jpg')]
                        max_frame = max(max_frame, int(os.path.splitext(image_filenames[-1])[0]) - 1)
                if self.mCurrentFrame > max_frame:
                    self.mCurrentFrame = max_frame
                image_filename = os.path.join(track_result, '{:05d}.jpg'.format(self.mCurrentFrame))
                self.mFrame = cv2.imread(image_filename)
                if self.mFrame is not None:
                    pixmap = convert_to_pixmap(self.mFrame)
                    video_window.setPixmap(pixmap)
                    video_window.setScaledContents(True)
            self.ui.horizontalSlider.setValue(int(self.mCurrentFrame))
            current_msec = int((self.mCurrentFrame / self.mFrameRate) * 1000)
            current_time = QTime(0, 0, 0, 0).addMSecs(current_msec)
            self.ui.timeLabel.setText(current_time.toString("hh:mm:ss"))
            self.ui.CurrentFrame.setText("Current Frame:{}".format(int(self.mCurrentFrame)))

    def pre_frame(self):
        self.theTimer.stop()
        self.setBroadcastState_BtnStyle()
        if self.mFrameRate == 0:
            pass
        else:
            self.mCurrentFrame -= 1
            if self.mCurrentFrame < 0:
                self.mCurrentFrame = 0
            for video_window, track_result in zip([self.ui.VideoWindow1, self.ui.VideoWindow2, self.ui.VideoWindow3],
                                                  [self.tracked_result1, self.tracked_result2, self.tracked_result3]):
                image_filename = os.path.join(track_result, '{:05d}.jpg'.format(self.mCurrentFrame))
                self.mFrame = cv2.imread(image_filename)
                if self.mFrame is not None:
                    pixmap = convert_to_pixmap(self.mFrame)
                    video_window.setPixmap(pixmap)
                    video_window.setScaledContents(True)
            self.ui.horizontalSlider.setValue(int(self.mCurrentFrame))
            current_msec = int((self.mCurrentFrame / self.mFrameRate) * 1000)
            current_time = QTime(0, 0, 0, 0).addMSecs(current_msec)
            self.ui.timeLabel.setText(current_time.toString("hh:mm:ss"))
            self.ui.CurrentFrame.setText("Current Frame:{}".format(int(self.mCurrentFrame)))

    def next_frame(self):
        self.theTimer.stop()
        self.setBroadcastState_BtnStyle()
        if self.mFrameRate == 0:
            pass
        else:
            self.mCurrentFrame += 1
            if self.mCurrentFrame > self.total_frames:
                self.mCurrentFrame = 0
            for video_window, track_result in zip([self.ui.VideoWindow1, self.ui.VideoWindow2, self.ui.VideoWindow3],
                                                  [self.tracked_result1, self.tracked_result2, self.tracked_result3]):
                image_filename = os.path.join(track_result, '{:05d}.jpg'.format(self.mCurrentFrame))
                self.mFrame = cv2.imread(image_filename)
                if self.mFrame is not None:
                    pixmap = convert_to_pixmap(self.mFrame)
                    video_window.setPixmap(pixmap)
                    video_window.setScaledContents(True)
            self.ui.horizontalSlider.setValue(int(self.mCurrentFrame))
            current_msec = int((self.mCurrentFrame / self.mFrameRate) * 1000)
            current_time = QTime(0, 0, 0, 0).addMSecs(current_msec)
            self.ui.timeLabel.setText(current_time.toString("hh:mm:ss"))
            self.ui.CurrentFrame.setText("Current Frame:{}".format(int(self.mCurrentFrame)))

    def Select_SORT(self):
        self.ALGOName_queue.enqueue("SORT")
        self.ui.ALGOname1.setText(self.ALGOName_queue.data[0])
        self.ui.ALGOname2.setText(self.ALGOName_queue.data[1])
        self.ui.ALGOname3.setText(self.ALGOName_queue.data[2])
        self.algo_list = self.ALGOName_queue.data
        self.ui.indicator_table.setVerticalHeaderLabels(self.algo_list)

    def Select_DeepSORT(self):
        self.ALGOName_queue.enqueue("DeepSORT")
        self.ui.ALGOname1.setText(self.ALGOName_queue.data[0])
        self.ui.ALGOname2.setText(self.ALGOName_queue.data[1])
        self.ui.ALGOname3.setText(self.ALGOName_queue.data[2])
        self.algo_list = self.ALGOName_queue.data
        self.ui.indicator_table.setVerticalHeaderLabels(self.algo_list)

    def Select_ByteTrack(self):
        self.ALGOName_queue.enqueue("ByteTrack")
        self.ui.ALGOname1.setText(self.ALGOName_queue.data[0])
        self.ui.ALGOname2.setText(self.ALGOName_queue.data[1])
        self.ui.ALGOname3.setText(self.ALGOName_queue.data[2])
        self.algo_list = self.ALGOName_queue.data
        self.ui.indicator_table.setVerticalHeaderLabels(self.algo_list)

    def Select_C_BIoUTrack(self):
        self.ALGOName_queue.enqueue("C-BIoUTrack")
        self.ui.ALGOname1.setText(self.ALGOName_queue.data[0])
        self.ui.ALGOname2.setText(self.ALGOName_queue.data[1])
        self.ui.ALGOname3.setText(self.ALGOName_queue.data[2])
        self.algo_list = self.ALGOName_queue.data
        self.ui.indicator_table.setVerticalHeaderLabels(self.algo_list)

    def Select_UAVMOT(self):
        self.ALGOName_queue.enqueue("UAVMOT")
        self.ui.ALGOname1.setText(self.ALGOName_queue.data[0])
        self.ui.ALGOname2.setText(self.ALGOName_queue.data[1])
        self.ui.ALGOname3.setText(self.ALGOName_queue.data[2])
        self.algo_list = self.ALGOName_queue.data
        self.ui.indicator_table.setVerticalHeaderLabels(self.algo_list)

    def Select_BotSORT(self):
        self.ALGOName_queue.enqueue("BotSORT")
        self.ui.ALGOname1.setText(self.ALGOName_queue.data[0])
        self.ui.ALGOname2.setText(self.ALGOName_queue.data[1])
        self.ui.ALGOname3.setText(self.ALGOName_queue.data[2])
        self.algo_list = self.ALGOName_queue.data
        self.ui.indicator_table.setVerticalHeaderLabels(self.algo_list)

    def Clear(self):
        self.ALGOName_queue.data = [None] * 3
        self.ALGOName_queue.head = 0
        self.ALGOName_queue.tail = 0
        self.ui.ALGOname1.setText(self.ALGOName_queue.data[0])
        self.ui.ALGOname2.setText(self.ALGOName_queue.data[1])
        self.ui.ALGOname3.setText(self.ALGOName_queue.data[2])
        self.algo_list = ['1', '2', '3']
        self.ui.indicator_table.setVerticalHeaderLabels(self.algo_list)

    def check_select_algo(self):
        if not self.ui.ALGOname1.text() or not self.ui.ALGOname2.text() or not self.ui.ALGOname3.text():
            self.msg_box.setIcon(QMessageBox.Warning)
            self.msg_box.setText("Please select three algorithms first")
            self.msg_box.setWindowTitle("Warning")
            self.msg_box.exec_()

    def set_table_style(self):
        self.ui.indicator_table.setRowCount(3)
        self.ui.indicator_table.setColumnCount(15)
        indicator_list = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe',
                          'AssPr', 'MOTA', 'MOTP', 'IDSW', 'MT', 'ML', 'IDF1',
                          'IDR', 'IDP']
        self.ui.indicator_table.setHorizontalHeaderLabels(indicator_list)
        self.ui.indicator_table.setStyleSheet("QTableWidget QTableCornerButton::section,QTableWidget "
                                              "QHeaderView::section, QTableWidget::item { border: 1px solid black; }")
        self.ui.indicator_table.horizontalHeader().setStretchLastSection(True)  # 单元横向高度自适应。铺满窗口
        self.ui.indicator_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.ui.indicator_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.ui.indicator_table.verticalHeader().setStretchLastSection(True)  # 单元竖直高度自适应。铺满窗口
        self.ui.indicator_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.ui.indicator_table.verticalHeader().setDefaultAlignment(Qt.AlignCenter)
        font = self.ui.indicator_table.font()
        font.setBold(True)
        font.setPointSize(12)
        self.ui.indicator_table.horizontalHeader().setFont(font)
        self.ui.indicator_table.verticalHeader().setFont(font)
        self.ui.indicator_table.setFont(QFont("song", 12))

    def read_indicator(self, data_dict, max_cols=[], min_cols=[]):
        for row in range(self.ui.indicator_table.rowCount()):
            row_header = self.ui.indicator_table.verticalHeaderItem(row).text()
            row_data = data_dict[row_header][row_header]
            for col in range(self.ui.indicator_table.columnCount()):
                col_header = self.ui.indicator_table.horizontalHeaderItem(col).text()
                item = QTableWidgetItem(str(row_data[col_header]))
                item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.ui.indicator_table.setItem(row, col, item)

        for col in max_cols:
            max_value = None
            for row in range(self.ui.indicator_table.rowCount()):
                item = self.ui.indicator_table.item(row, col)
                if item is not None:
                    value = float(item.text())
                    if max_value is None or value > max_value:
                        max_value = value
            for row in range(self.ui.indicator_table.rowCount()):
                item = self.ui.indicator_table.item(row, col)
                if item is not None:
                    value = float(item.text())
                    if value == max_value:
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
        for col in min_cols:
            min_value = None
            for row in range(self.ui.indicator_table.rowCount()):
                item = self.ui.indicator_table.item(row, col)
                if item is not None:
                    value = float(item.text())
                    if min_value is None or value < min_value:
                        min_value = value
            for row in range(self.ui.indicator_table.rowCount()):
                item = self.ui.indicator_table.item(row, col)
                if item is not None:
                    value = float(item.text())
                    if value == min_value:
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)

    def setBroadcastState_BtnStyle(self):
        style = "QPushButton { \
            border: 0px; \
            image: url(:/broadcastBtn_normal.png); \
        } \
        QPushButton:hover { \
            image: url(:/broadcastBtn_hover.png); \
        } \
        QPushButton:pressed { \
            image: url(:/broadcastBtn_press.png); \
        }"
        self.ui.broadcastBtn.setStyleSheet(style)

    def setPauseState_BtnStyle(self):
        style = "QPushButton { \
            border: 0px; \
            image: url(:/pauseBtn_normal.png); \
        } \
        QPushButton:hover { \
            image: url(:/pauseBtn_hover.png); \
        } \
        QPushButton:pressed { \
            image: url(:/pauseBtn_press.png); \
        }"
        self.ui.broadcastBtn.setStyleSheet(style)

    def setStop_BtnStyle(self):
        style = "QPushButton { \
            border:0px;\
            image: url(:/stopBtn_normal.png);\
        }\
        QPushButton:hover {\
            image: url(:/stopBtn_normal.png);\
        }\
        QPushButton:pressed {\
            image: url(:/stopBtn_press.png);\
        }"
        self.ui.stopBtn.setStyleSheet(style)

    def setCentral_widgetStyle(self):
        style = "#centralwidget{ \
            border-image: url(:/background_1.png);\
            }"
        self.ui.centralwidget.setStyleSheet(style)

    def setStatus_barStyle(self):
        style = "#statusbar{ \
            border-image: url(:/background.png);\
            }"
        self.ui.statusbar.setStyleSheet(style)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow()
    w.ui.setWindowTitle("Multi-target detection and tracking comparison system")
    center(w)  # Move the window to a comfortable position
    w.ui.show()
    app.exec()
