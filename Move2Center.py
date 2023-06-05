from PyQt5.QtWidgets import QDesktopWidget

#Move the window to a comfortable position
def center(w):
    screen = QDesktopWidget().screenGeometry()     #Get screen size
    size = w.ui.geometry()                         #Get ui size
    w.ui.move(int((screen.width() - size.width()) / 2), int((screen.height() - size.height()) / 2)-70)