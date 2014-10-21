import sys
import os
import time
import datetime
import logging
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import zmq
from skimage import color
from network import recv_array
reload(sys)
sys.setdefaultencoding("utf-8")


COLORS = matplotlib.rcParams["axes.color_cycle"]


class AppForm(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.setWindowTitle("Labeler")

        logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s "
                            "- %(message)s", datefmt="%Y/%m/%d %H:%M:%S",
                            level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.context = zmq.Context()
        self.sock = self.context.socket(zmq.REQ)
        self.sock.connect("tcp://192.168.178.27:5678")

        self.create_menu()
        self.create_main_frame()
        self.create_status_bar()

        self.logger.info("Initialization done.")

    def create_main_frame(self):
        self.main_frame = QWidget()

        self.fig = Figure((8.0, 4.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        self.axis = self.fig.add_subplot(111)
        self.axis.set_xticks(())
        self.axis.set_yticks(())
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        self.download_button = QPushButton("Download")
        self.connect(self.download_button, SIGNAL('clicked()'), self.on_download)

        self.label_button_1 = QPushButton("&Label 1")
        self.connect(self.label_button_1, SIGNAL('clicked()'), self.on_label_1)

        self.label_button_2 = QPushButton("&Label 2")
        self.connect(self.label_button_2, SIGNAL('clicked()'), self.on_label_2)

        hbox = QHBoxLayout()

        for w in [self.download_button, self.label_button_1, self.label_button_2]:
            hbox.addWidget(w)
            hbox.setAlignment(w, Qt.AlignVCenter)

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addLayout(hbox)

        main_layout = QHBoxLayout()
        main_layout.addLayout(vbox)

        self.main_frame.setLayout(main_layout)
        self.setCentralWidget(self.main_frame)

    def create_menu(self):        
        self.file_menu = self.menuBar().addMenu(self.tr("&File"))

        quit_action = self.create_action(
            self.tr("&Quit"), slot=self.close, shortcut="Ctrl+Q",
            tip=self.tr("Close the application"))

        self.add_actions(self.file_menu, (quit_action,))

    def create_status_bar(self):
        self.status_text = QLabel("")
        self.statusBar().addWidget(self.status_text, 1)

    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def create_action(self, text, slot=None, shortcut=None, icon=None,
                      tip=None, checkable=False, signal="triggered()"):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            self.connect(action, SIGNAL(signal), slot)
        if checkable:
            action.setCheckable(True)
        return action

    def on_download(self):
        self.axis.clear()
        self.axis.set_xticks(())
        self.axis.set_yticks(())
        self.sock.send("")
        self.image = recv_array(self.sock)
        #gray = color.rgb2gray(self.image)
        self.axis.imshow(self.image)
        self.canvas.draw()

    def on_label_1(self):
        self.on_label(0)

    def on_label_2(self):
        self.on_label(1)

    def on_label(self, label):
        print("Label %d" % label)
        timestamp = "%d" % time.time()
        np.save("data/%s.npy" % timestamp, self.image)
        f = open("data/labels.txt", "a")
        f.write(timestamp + " " + str(label) + "\n")


def main():
    app = QApplication(sys.argv)
    locale = QLocale.system().name()
    qt_translator = QTranslator()
    if qt_translator.load("qt_" + locale):
        app.installTranslator(qt_translator)
    form = AppForm()
    form.show()
    app.exec_()


if __name__ == "__main__":
    main()
