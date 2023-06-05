from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox


class CircularQueue:
    def __init__(self, k: int):
        self.size = k
        self.data = [None] * k
        self.head = 0
        self.tail = 0
        self.msg_box = QMessageBox()
        self.msg_box.setWindowIcon(QIcon("./pic/logo.png"))

    def enqueue(self, value: str) -> bool:
        if self.isFull():
            if value in self.data:
                index = self.data.index(value)
                self.data[0], self.data[index] = self.data[index], self.data[0]
                self.head = (self.head + 1) % self.size
                self.tail = (self.tail + 1) % self.size
                return True
            else:
                self.head = (self.head + 1) % self.size
        if value in self.data:
            self.msg_box.setIcon(QMessageBox.Critical)
            self.msg_box.setText("This algorithm has been selected")
            self.msg_box.setWindowTitle("Warning")
            self.msg_box.exec_()
            return
        else:
            self.data[self.tail] = value
            self.tail = (self.tail + 1) % self.size
            return True

    def dequeue(self) -> bool:
        if self.isEmpty():
            return False

        self.head = (self.head + 1) % self.size
        return True

    def Front(self) -> str:
        return str(self.data[self.head])

    def Rear(self) -> str:
        return str(self.data[self.tail - 1])

    def isEmpty(self) -> bool:
        return self.head == self.tail and self.data[self.head] == -1

    def isFull(self) -> bool:
        return self.head == self.tail and self.data[self.head] != -1 and None not in self.data
