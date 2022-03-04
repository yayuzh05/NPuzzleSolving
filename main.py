#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy.core._dtype_ctypes
import sys
import threading
import cv2
from PyQt5.QtWidgets import QMainWindow, QWidget, QGridLayout, QPushButton, QHBoxLayout, QVBoxLayout, QApplication, \
    QLineEdit, QLabel, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMovie, QPainter, QImage, QPixmap

# initialization
import random
import time
import numpy as np
from PyQt5.uic.Compiler.qtproxies import QtCore


def matrixRandomization(m, n):
    a = [x for x in range(0, m*n)]
    random.shuffle(a)
    matrix = np.array(a).reshape(m, n)
    return matrix


def inverseNumber(matrix):
    ans = 0
    (m,n) = matrix.shape
    mat = matrix.reshape(1,m*n)
    for i in range(m*n):
        for j in range(i):
            if mat[0,j] > mat[0,i] and mat[0,i] and mat[0,j]:
                ans += 1
    return ans


def judgeAnswer(initial_in, target_in):
    (m,n) = initial_in.shape
    comp1 = inverseNumber(initial_in) % 2
    comp2 = inverseNumber(target_in) % 2
    if n % 2 == 1:
        if comp1 == comp2:
            return 1
        else:
            return 0
    else:
        l1 = np.argwhere(initial_in == 0)
        l2 = np.argwhere(target_in == 0)
        if (abs(l1[0][0] - l2[0][0]) % 2 + abs(comp1 - comp2)) % 2 == 0:
            return 1
        else:
            return 0


def matrixOutput(m, n, target_in):
    initial_in = matrixRandomization(m,n)
    while judgeAnswer(initial_in, target_in) == 0:
        initial_in = matrixRandomization(m,n)
    return initial_in


def costFunction(temp, target):
    p = 0
    s = 0
    (m, n) = temp.shape
    flag = (m % 2 == 1 and n % 2 == 1)
    l1 = np.argwhere(temp == 0)
    l2 = np.argwhere(target == 0)
    for i in temp:
        for x in i:
            if x != 0:
                comp1 = 0
                comp2 = 0
                list1 = np.argwhere(x == temp)
                list2 = np.argwhere(x == target)
                (a, b) = list1[0]
                (c, d) = list2[0]
                p = p + abs(a - c) ** 2 + abs(b - d) ** 2 + (abs(a - c) > 0) * 2 + (abs(b - d) > 0) * 2
                if a == m / 2 and b == n / 2 and flag:
                    s = s + 1
                    continue
                elif b == n - 1:
                    if d != n - 1:
                        if target[c][d + 1] != 0:
                            s = s + 2
                        elif d != n - 2:
                            s = s + 2
                    continue
                elif d == n - 1:
                    if temp[a][b + 1] != 0:
                        s = s + 2
                    elif b != n - 2:
                        s = s + 2
                    continue
                elif temp[a][b + 1] == 0:
                    if b != n - 2:
                        comp1 = temp[a][b + 2]
                        if target[c][d + 1] == 0:
                            if d != n - 2:
                                comp2 = target[c][d + 2]
                            else:
                                s = s + 2
                                continue
                    elif target[c][d + 1] != 0:
                        s = s + 2
                        continue
                    elif d != n - 2:
                        s = s + 2
                        continue
                elif target[c][d + 1] == 0:
                    if d != n - 2:
                        comp2 = target[c][d + 2]
                    elif temp[a][b + 1] != 0:
                        s = s + 2
                        continue
                    elif b != n - 2:
                        s = s + 2
                        continue
                if comp1 == 0:
                    comp1 = temp[a][b + 1]
                if comp2 == 0:
                    comp2 = target[c][d + 1]
                if comp2 != comp1:
                    s = s + 2
    return 3 * p + 9 * s


def takeSecond(elem):
    return elem[1]


def findSpace(matrix):
    a = np.argwhere(matrix == 0)
    return [a[0, 0], a[0, 1]]


def moveZero(temp, a):
    mat = np.array(temp[0], copy=True)  # 1 for up, 2 for down, 3 for left, 4 for right
    if a == 3:
        mat[temp[2][0], temp[2][1]] = mat[temp[2][0], temp[2][1] - 1]
        mat[temp[2][0], temp[2][1] - 1] = 0
    elif a == 4:
        mat[temp[2][0], temp[2][1]] = mat[temp[2][0], temp[2][1] + 1]
        mat[temp[2][0], temp[2][1] + 1] = 0
    elif a == 1:
        mat[temp[2][0], temp[2][1]] = mat[temp[2][0] - 1, temp[2][1]]
        mat[temp[2][0] - 1, temp[2][1]] = 0
    elif a == 2:
        mat[temp[2][0], temp[2][1]] = mat[temp[2][0] + 1, temp[2][1]]
        mat[temp[2][0] + 1, temp[2][1]] = 0
    return mat


def findSuccessor(temp, m, n):
    successor = []
    if temp[2][0] == m - 1:
        if temp[2][1] == n - 1:
            successor.extend([moveZero(temp, 1), moveZero(temp, 3)])  # 1 for up, 2 for down, 3 for left, 4 for right
        elif temp[2][1] == 0:
            successor.extend([moveZero(temp, 1), moveZero(temp, 4)])
        else:
            successor.extend([moveZero(temp, 1), moveZero(temp, 3), moveZero(temp, 4)])

    elif temp[2][0] == 0:
        if temp[2][1] == n - 1:
            successor.extend([moveZero(temp, 2), moveZero(temp, 3)])
        elif temp[2][1] == 0:
            successor.extend([moveZero(temp, 2), moveZero(temp, 4)])
        else:
            successor.extend([moveZero(temp, 2), moveZero(temp, 3), moveZero(temp, 4)])
    else:
        if temp[2][1] == n - 1:
            successor.extend([moveZero(temp, 1), moveZero(temp, 2), moveZero(temp, 3)])
        elif temp[2][1] == 0:
            successor.extend([moveZero(temp, 1), moveZero(temp, 2), moveZero(temp, 4)])
        else:
            successor.extend([moveZero(temp, 1), moveZero(temp, 2), moveZero(temp, 3), moveZero(temp, 4)])
    return successor


def sudoku(initial_in, target_in):
    step = 0
    node = 0
    (m, n) = initial_in.shape
    initial = [initial_in, costFunction(initial_in, target_in), findSpace(initial_in), step]
    openList = [initial]
    pathList = []
    closedDict = {}
    fatherDict = {}
    while openList:
        openList.sort(key=takeSecond)
        temp = openList[0]
        # print(temp[0])
        # print('最小f：', temp[1], ' 已扩展节点：',  node)
        openList.pop(0)
        closedDict[str(temp[0])] = temp[1]
        if len(np.argwhere(temp[0] != target_in)) == 0: # 可改进
            initial_str = str(initial_in)
            temp_str = str(temp[0])
            pathList.insert(0, temp[0])
            while temp_str != initial_str:
                pathList.insert(0, fatherDict[temp_str][1])
                temp_str = fatherDict[temp_str][0]
            return pathList
        else:
            successor = findSuccessor(temp, m, n)
            for x in successor:
                if str(x) not in closedDict:
                    openList.append([x, temp[3]+1+costFunction(x, target_in), findSpace(x), temp[3]+1])
                    fatherDict[str(x)] = (str(temp[0]), temp[0])
                    node += 1


def findDirection(t):
    l = len(t)
    direList = []
    for i in range(l-1):
        list1 = np.argwhere(t[i] == 0)
        list2 = np.argwhere(t[i+1] == 0)
        (a1,b1) = list1[0]
        (a2,b2) = list2[0]
        if abs(a1 - a2):
            if a1 > a2:
                direList.append((t[i][a2][b2],2))  # 1 for up, 2 for down, 3 for left, 4 for right
            else:
                direList.append((t[i][a2][b2],1))
        elif b1 - b2 > 0:
            direList.append((t[i][a2][b2],4))
        else:
            direList.append((t[i][a2][b2],3))
    return direList


class PicMoveThread(threading.Thread):
    def __init__(self, pic, s):
        super().__init__()
        self.pic = pic
        self.s = s

    def run(self):
        self.pic.easyMove(self.s)


class NumMoveThread(threading.Thread):
    def __init__(self, num, t, height, width):
        super().__init__()
        self.num = num
        self.t = t
        self.height = height
        self.width = width

    def run(self):
        for x in self.t:
            time.sleep(1)
            self.num.changeNum(self.height, self.width, x)


class SolveThread(threading.Thread):

    def __init__(self, parent,  initial, target):
        super().__init__()
        self.initial = initial
        self.target = target

    def run(self):
        s = sudoku(self.initial, self.target)
        # t = findDirection(self.tool.buffer1)
        self.solveSign.emit(len(self.tool.buffer))


class PicBlock(QWidget):

    def __init__(self, parent, number, state, height, width, img=None):
        super().__init__(parent)
        self.number = number
        hlayout = QHBoxLayout()
        hlayout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(hlayout)
        if state == 1:      # Show in number
            label = QLabel(str(number))
            label.setAlignment(Qt.AlignCenter)
            hlayout.addWidget(label)
        else:    # Show in pic
            img = cv2.resize(img, (width, height))
            # img = np.array(img).astype(np.int32)
            showImage = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
            label = QLabel()
            label.setPixmap(QPixmap(showImage))
            hlayout.addWidget(label)
        self.setFixedSize(width, height)


class PicDemo(QWidget):

    def __init__(self, label):
        super().__init__()
        self.widgets = []
        self.initial = {}
        self.img = cv2.imread('background.png', 1)
        self.setFixedSize(600, 600)
        self.h = 200
        self.wid = 200
        self.state = 0
        self.label = label
        file = open('style1.qss', 'r')
        self.setStyleSheet(file.read())

    def reset(self, height, width, number, target=None):
        for tmp in self.widgets:
            if tmp:
                tmp.setParent(None)
        self.h = self.height() // height
        self.wid = self.width() // width
        self.widgets = []
        pic_h = self.img.shape[0] // height
        pic_w = self.img.shape[1] // width
        for i in range(height):
            for j in range(width):
                self.initial[number[i][j]] = i * width + j
                if target is not None:
                    loc = np.argwhere(target == number[i][j])[0]
                    i_1 = loc[0]
                    j_1 = loc[1]
                    sub = self.img[i_1*pic_h:(i_1+1)*pic_h, j_1*pic_w:(j_1+1)*pic_w, :]
                else:
                    sub = self.img[i*pic_h:(i+1)*pic_h, j*pic_w:(j+1)*pic_w, :]
                tmp = PicBlock(self, number[i][j], self.state, self.h, self.wid, sub)
                tmp.show()
                if number[i][j] == 0:
                    tmp.setHidden(1)
                tmp.move(j * self.wid, i * self.h)
                self.widgets.append(tmp)
        # for i in range(height*width):
            # (self.widgets[self.initial[i]].number)

    def easyMove(self, s):
        for x in s:
            tmp = self.widgets[self.initial[x[0]]]
            if x[1] == 1:
                initial = (tmp.x(), tmp.y())
                delta = self.h / 20
                for i in range(20):
                    time.sleep(0.1/30)
                    tmp.move(initial[0], initial[1] - (i+1) * delta)
                self.label.setText('当前移动：'+str(tmp.number) + ' 向上')
            elif x[1] == 2:
                initial = (tmp.x(), tmp.y())
                delta = self.h / 20
                for i in range(20):
                    time.sleep(0.1 / 30)
                    tmp.move(initial[0], initial[1] + (i+1) * delta)
                self.label.setText('当前移动：'+str(tmp.number) + ' 向下')
            elif x[1] == 3:
                initial = (tmp.x(), tmp.y())
                delta = self.wid / 20
                for i in range(20):
                    time.sleep(0.1 / 30)
                    tmp.move(initial[0] - (i+1) * delta, initial[1])
                self.label.setText('当前移动：'+str(tmp.number) + ' 向左')
            else:
                initial = (tmp.x(), tmp.y())
                delta = self.wid / 20
                for i in range(20):
                    time.sleep(0.1 / 30)
                    tmp.move(initial[0] + (i + 1) * delta, initial[1])
                self.label.setText('当前移动：'+str(tmp.number) + ' 向右')
        # for tmp in self.widgets:
            # tmp.setVisible(1)
        self.label.setText('准备就绪')


class NumDemo(QWidget):
    def __init__(self, width, height, number):
        super().__init__()
        self.grid = QGridLayout()
        self.setFixedSize(300, 300)
        self.setLayout(self.grid)
        self.myWidget = []
        self.setState(width, height, number)

    def setState(self, height, width, number):
        for tmp in self.myWidget:
            if tmp:
                tmp.setParent(None)
        self.myWidget = []
        for i in range(height):
            for j in range(width):
                edit = QLineEdit(str(number[i][j]))
                self.grid.addWidget(edit, *(i, j))
                self.myWidget.append(edit)

    def changeNum(self, height, width, number):
        for i in range(height):
            for j in range(width):
                self.myWidget[i * height + j].setText(str(number[i][j]))
                self.myWidget[i * height + j].repaint()
                # print(number[i][j])
        # self.grid.update();

    def getState(self, height, width):
        num = []
        for i in range(height):
            for j in range(width):
                num.append(int(self.myWidget[i * height + j].text()))
        mat = np.array(num).reshape(height, width)
        return mat


class Tool(QWidget):
    def __init__(self, pic, label):
        super().__init__()
        self.buffer1 = []
        self.buffer2 = []
        self.pic = pic
        self.width = 3
        self.height = 3
        self.target = self.getTarget()
        self.number = self.target
        self.pic.reset(self.height, self.width, self.number)
        self.label = label
        self.setFixedSize(300, 700)
        self.setupUi()
        file = open('style2.qss', 'r')
        self.setStyleSheet(file.read())

    def setupUi(self):
        vlayout = QVBoxLayout()
        self.setLayout(vlayout)
        editlayout1 = QHBoxLayout();
        editlayout2 = QHBoxLayout();
        editlayout3 = QVBoxLayout();
        self.wLabel = QLabel('Col Num:')
        self.hLabel = QLabel('Row Num:')
        self.wEdit = QLineEdit('3')
        self.hEdit = QLineEdit('3')
        editlayout1.addWidget(self.wLabel)
        editlayout1.addWidget(self.wEdit)
        editlayout2.addWidget(self.hLabel)
        editlayout2.addWidget(self.hEdit)
        self.num = NumDemo(self.height, self.width, self.number)
        self.solveButton = QPushButton('Solve')
        self.solveButton.clicked.connect(self.solve)
        self.stateButton = QPushButton('Randomize')
        self.stateButton.clicked.connect(self.getRandomized)
        self.updateButton = QPushButton('Dim Update')
        self.updateButton.clicked.connect(self.getDimUpdated)
        self.gridButton = QPushButton('Matrix Update')
        self.gridButton.clicked.connect(self.getMatUpdated)
        editlayout3.addWidget(self.num)
        editlayout3.addWidget(self.solveButton)
        editlayout3.addWidget(self.stateButton)
        editlayout3.addWidget(self.updateButton)
        editlayout3.addWidget(self.gridButton)
        vlayout.addLayout(editlayout1)
        vlayout.addLayout(editlayout2)
        vlayout.addLayout(editlayout3)

    def solve(self):
        self.label.setText('搜索中...')
        t = sudoku(self.number, self.target)
        s = findDirection(t)
        # solve.solveSign.connect(self.drawPath)
        thread1 = PicMoveThread(self.pic, s)
        thread1.start()
        self.number = self.target
        self.num.setState(self.height, self.width, self.number)
        # thread2 = NumMoveThread(self.num, t, self.height, self.width)
        # thread2.start()

    def drawPath(self):
        newThread1 = PicMoveThread(self.pic, self.buffer2)
        # newThread2 = NumMoveThread(self.num, t, self.height, self.width)
        newThread1.start()
        # QtWidgets.QApplication.processEvents()

    def getRandomized(self):
        self.number = matrixOutput(self.height, self.width, self.target)
        self.num.setState(self.height, self.width, self.number)
        self.pic.reset(self.height, self.width, self.number, self.target)
        return self.number

    def getParameter(self):
        ''' if (self.solving):
            return False '''
        try:
            if self.wEdit.text() != '':
                self.width = int(self.wEdit.text())
            if self.hEdit.text() != '':
                self.height = int(self.hEdit.text())
            # self.speed = float(self.speedEdit.edit.text())
            return True

        except:
            QMessageBox.information(self, 'Error', 'Data Error.')
            return False

    def getDimUpdated(self):
        if not self.getParameter():
            return
        self.target = self.getTarget()
        self.number = self.target
        self.pic.reset(self.height, self.width, self.number, self.target)
        self.num.setState(self.height, self.width, self.number)

    def getMatUpdated(self):
        self.number = self.num.getState(self.height, self.width)
        self.pic.reset(self.height, self.width, self.number, self.target)
        self.num.setState(self.height, self.width, self.number)

    def getTarget(self):
        if self.width == 3 and self.height == 3:
            return np.array(([1,2,3],[8,0,4],[7,6,5]))
        elif self.width == 4 and self.height == 4:
            return np.array(([1,2,3,4],[12,13,14,5],[11,0,15,6],[10,9,8,7]))
        elif self.width == 5 and self.height == 5:
            return np.array(([1,2,3,4,5],[16,17,18,19,6],[15,24,0,20,7],[14,23,22,21,8],[13,12,11,10,9]))
        elif self.width == 6 and self.height == 6:
            return np.array(([1,2,3,4,5,6],[20,21,22,23,24,7],[19,32,33,34,25,8],[18,31,0,35,26,9],[17,30,29,28,27,10],[16,15,14,13,12,11]))
        else:
            return matrixRandomization(self.height, self.width)

    def setNum(self):
        self.pic.state = 1
        self.pic.reset(self.height, self.width, self.number, self.target)

    def setPic(self):
        self.pic.state = 0
        self.pic.reset(self.height, self.width, self.number, self.target)


class MainContent(QWidget):
    def __init__(self):
        super().__init__()
        mainlayout = QVBoxLayout()
        self.setLayout(mainlayout)
        hlayout1 = QHBoxLayout()
        hlayout2 = QHBoxLayout()
        self.label = QLabel('准备就绪')
        self.pic = PicDemo(self.label)
        self.tool = Tool(self.pic, self.label)
        self.numButton = QPushButton('Set Number')
        self.picButton = QPushButton('Set Image')
        self.numButton.clicked.connect(self.tool.setNum)
        self.picButton.clicked.connect(self.tool.setPic)
        hlayout1.addWidget(self.pic)
        hlayout1.addWidget(self.tool)
        hlayout2.addWidget(self.label)
        hlayout2.addWidget(self.numButton)
        hlayout2.addWidget(self.picButton)
        mainlayout.addLayout(hlayout1)
        mainlayout.addLayout(hlayout2)


class MainWindow(QMainWindow, QLabel):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('重排九宫')
        self.setCentralWidget(MainContent())
        self.movie = QMovie("background3.gif")
        self.movie.frameChanged.connect(self.repaint)
        self.movie.start()
        file = open('style.qss', 'r')
        self.setStyleSheet(file.read())

    def paintEvent(self, event):
        currentFrame = self.movie.currentPixmap()
        frameRect = currentFrame.rect()
        frameRect.moveCenter(self.rect().center())
        if frameRect.intersects(event.rect()):
            painter = QPainter(self)
            painter.drawPixmap(frameRect.left(), frameRect.top(), currentFrame)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
