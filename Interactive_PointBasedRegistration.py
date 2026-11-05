# % Class to perform interactive rigid or elastic point based registration
# % ECE 4370: Engineering for Surgery
# % Fall 2024
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu

from volumeViewer import *
import numpy as np
from matplotlib.widgets import Button
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt

# from mayavi import mlab
import matplotlib.text as txt
from RigidPointRegister import *
from Project4_2 import imageProjection
from ThinPlateSpline import *
from vtkWin import *

class evnt:
    def __init__(self):
        self.dblclick=True
        self.button = MouseButton.LEFT
        self.inaxes = []
        self.xdata = []
        self.ydata = []

class ipbrwin:
    def __init__(self, img, voxsz, contrast, level, landmarks=None, RigidRegistration=None, ThinPlateSpline=None):
        self.mode = 0
        self.finished = 0
        self.d = volumeViewer('Interactive Point-Based Registration')
        self.d.setImage(img, voxsz, contrast=contrast, level=level)
        self.d2 = None
        self.curim=0
        self.img1 = img
        self.img2 = []
        self.contrast1 = contrast
        self.level1 = level
        self.contrast2 = []
        self.level2 = []
        for i in range(0,3):
            self.d.update(direction=i)

        self.btns = []
        self.AddButtons(RigidRegistration, ThinPlateSpline)
        if landmarks is None:
            self.landmarks = []
        else:
            self.landmarks = landmarks
        self.curlandmark = -1

        plt.axes([0.45, 0.03, 0.05, 0.45])
        plt.axis('off')
        self.ht = plt.text(0, .1, '*')
        self.ht.set(visible=False)

        plt.axes([0.75, 0.03, 0.25, 0.45])
        plt.axis('off')
        self.ft = np.zeros(16, dtype=txt.Text)
        self.ft[0] = plt.text(0.1, 1, '0 Fiducials:')
        self.ft[0].set(visible=True)
        for i in range(15):
            self.ft[i+1] = plt.text(0.1, 1-(i+1)/15, '')
        if len(self.landmarks) > 0:
            self.UpdatePointList()

        plt.connect('button_press_event', self.on_mouse_click)
        plt.connect('key_press_event', self.on_key_press)

    def setd2(self, d2):
        self.d2 = d2

    def AddButtons(self, RigidRegistration, ThinPlateSpline):
        lpos = .5 # .55
        ax1 = plt.axes([lpos, 0.4, 0.25, 0.05])
        self.btns.append(Button(ax1, 'Add Fiducial'))
        self.btns[-1].on_clicked(self.AddFiducial)

        ax2 = plt.axes([lpos, 0.325, 0.25, 0.05])
        self.btns.append(Button(ax2, 'Review Fiducials'))
        self.btns[-1].on_clicked(self.ReviewFiducials)

        ax3 = plt.axes([lpos, 0.25, 0.25, 0.05])
        self.btns.append(Button(ax3, 'Remove Fiducial'))
        self.btns[-1].on_clicked(self.RemoveFiducial)

        ax4 = plt.axes([lpos, 0.175, 0.25, 0.05])
        self.btns.append(Button(ax4, 'Rigid Register'))
        self.btns[-1].on_clicked(RigidRegistration)

        ax5 = plt.axes([lpos, 0.1, 0.25, 0.05])
        self.btns.append(Button(ax5, 'TPS Register'))
        self.btns[-1].on_clicked(ThinPlateSpline)

        ax6 = plt.axes([lpos, 0.025, 0.25, 0.05])
        self.btns.append(Button(ax6, 'Exit'))
        self.btns[-1].on_clicked(self.Exit)

    def AddFiducial(self, event):
        self.mode=1
        self.ht.set_y(0.85)
        self.ht.set(visible=True)

    def ReviewFiducials(self, event):
        self.mode = 2
        self.ht.set_y(0.7)
        self.ht.set(visible=True)

    def RemoveFiducial(self, event):
        self.mode = 0
        self.ht.set_y(0.55)
        self.ht.set(visible=True)
        if self.curlandmark<0:
            self.landmarks.pop()
        else:
            self.landmarks.pop(self.curlandmark)
            self.curlandmark -= 1
        self.UpdatePointList()

    def UpdatePointList(self):
        str = f'{np.shape(self.landmarks)[0]} Fiducials:'
        self.ft[0].set_text(str)
        for i in range(np.min([np.shape(self.landmarks)[0], 15])):
            str = f'\n{i}: {self.landmarks[i][0]:.1f} {self.landmarks[i][1]:.1f} {self.landmarks[i][2]:.1f}'
            self.ft[i+1].set_text(str)
            if i==self.curlandmark:
                self.ft[i + 1].set_fontweight('bold')
            else:
                self.ft[i+1].set_fontweight('normal')

        for i in range(np.shape(self.landmarks)[0], 15):
            self.ft[i + 1].set_text('')

        # if usemayavi:
        #     mlab.figure(self.d.f3d)
        #     mlab.clf()
        #     for i in range(np.shape(self.landmarks)[0]):
        #         mlab.points3d(self.landmarks[i][0], self.landmarks[i][1], self.landmarks[i][2], color=(1.,0.,0.), scale_factor=4.0)




    def Exit(self,event):
        self.finished=1

    def on_mouse_click(self, event):
        if self.mode==1 and event.button == MouseButton.LEFT:
            win=-1
            for i in range(0, 3):
                if event.inaxes == self.d.ax[i // 2, i % 2]:
                    win = i
                    break
            if win<0:
                return
            elif win == 2:
                pnt = [event.xdata, event.ydata, self.d.slc[2]]
            elif win == 1:
                pnt = [event.xdata, self.d.slc[1], event.ydata]
            elif win == 0:
                pnt = [self.d.slc[0], event.xdata, event.ydata]
            else:
                return

            self.landmarks.append(pnt)
            self.UpdatePointList()

        elif event.button == MouseButton.RIGHT:
            if np.size(self.img2)>0:
                if self.curim == 0:
                    self.contrast1 = self.d.contrast
                    self.level1 = self.d.level
                else:
                    self.contrast2 = self.d.contrast
                    self.level2 = self.d.level
                self.curim = (self.curim + 1)%2
                if self.curim == 0:
                    self.d.img = self.img1
                    self.d.contrast = self.contrast1
                    self.d.level = self.level1
                else:
                    self.d.img = self.img2
                    self.d.contrast = self.contrast2
                    self.d.level = self.level2

                self.d.update()

    def on_key_press(self, event):
        if self.mode!=2 or not (event.key == 'left' or event.key == 'right'):
            return
        if self.curlandmark<0:
            self.curlandmark=0
        elif event.key == 'left':
            self.curlandmark = (self.curlandmark - 1)%np.shape(self.landmarks)[0]
        elif event.key == 'right':
            self.curlandmark = (self.curlandmark + 1) % np.shape(self.landmarks)[0]
        self.UpdatePointList()

        e = evnt()
        e.inaxes = event.inaxes
        if e.inaxes == self.d.ax[0,0]:
            e.xdata = self.landmarks[self.curlandmark][1]
            e.ydata = self.landmarks[self.curlandmark][2]
        elif e.inaxes == self.d.ax[0,1]:
            e.xdata = self.landmarks[self.curlandmark][0]
            e.ydata = self.landmarks[self.curlandmark][2]
        elif e.inaxes == self.d.ax[1, 0]:
            e.xdata = self.landmarks[self.curlandmark][0]
            e.ydata = self.landmarks[self.curlandmark][1]
        self.d.slc = np.round(self.landmarks[self.curlandmark][:]).astype(np.longlong)
        self.d.slc[self.d.slc<0] = 0
        for i in (0, 1, 2):
            if np.shape(self.d.img)[i] <= self.d.slc[i]:
                self.d.slc[i] = np.shape(self.d.img)[i] - 1
        self.d.onMouseClick(e)


class interactive_pointBasedRegistration:
    def __init__(self, img1, voxsz1,img2, voxsz2, contrast1=1000, level1=0,
                 contrast2=1000, level2=0, landmarks1 = None, landmarks2 = None):
        self.finished = 0
        if landmarks1 is None:
            landmarks1 = []
        if landmarks2 is None:
            landmarks2 = []
        self.win1 = ipbrwin(img1, voxsz1, contrast1, level1, landmarks1, self.RigidRegistration, self.ThinPlateSpline)
        self.win2 = ipbrwin(img2, voxsz2, contrast2, level2, landmarks2, self.RigidRegistration, self.ThinPlateSpline)
        self.win1.d2 = self.win2.d
        self.win2.d2 = self.win1.d

        self.T1to2 = []
        self.D = []
        self.Db = []
        while self.win1.finished==0 and self.win2.finished==0:
            self.win1.d.fig.canvas.draw_idle()
            self.win1.d.fig.canvas.start_event_loop(0.3)

        # mlab.close(self.win1.d.f3d)
        # plt.close(self.win1.d.fig)
        # mlab.close(self.win2.d.f3d)
        # plt.close(self.win2.d.fig)

    def RigidRegistration(self, event):
        if np.size(self.win1.landmarks) != np.size(self.win2.landmarks):
            print('Error: Unequal number of landmarks')
            return

        l1 = np.array(self.win1.landmarks) * \
             np.repeat(self.win1.d.voxsz[np.newaxis,:],np.shape(self.win1.landmarks)[0], axis=0)
        l2 = np.array(self.win2.landmarks) * \
             np.repeat(self.win2.d.voxsz[np.newaxis, :], np.shape(self.win2.landmarks)[0], axis=0)
        p1to2, self.T1to2 = rigidPointRegister(l1,l2)
        self.win1.img2 = imageProjection(self.win2.d.img, np.shape(self.win1.d.img), sourcevoxsz=self.win2.d.voxsz,
                                 targetvoxsz=self.win1.d.voxsz, T=self.T1to2)
        self.win1.contrast2 = self.win2.d.contrast
        self.win1.level2 = self.win2.d.level
        self.win2.img2 = imageProjection(self.win1.d.img, np.shape(self.win2.d.img), sourcevoxsz=self.win1.d.voxsz,
                        targetvoxsz=self.win2.d.voxsz, T=np.linalg.inv(self.T1to2))
        self.win2.contrast2 = self.win1.d.contrast
        self.win2.level2 = self.win1.d.level


    def ThinPlateSpline(self, event):
        if np.size(self.win1.landmarks) != np.size(self.win2.landmarks):
            print('Error: Unequal number of landmarks')
            return

        l1 = np.array(self.win1.landmarks)
        l2 = np.array(self.win2.landmarks)
        Dx, Dy, Dz = thinPlateSpline(l2, l1, np.shape(self.win1.d.img))
        Dx2, Dy2, Dz2 = thinPlateSpline(l1, l2, np.shape(self.win2.d.img))
        self.D = np.concatenate((Dx[np.newaxis,:,:,:],Dy[np.newaxis,:,:,:],Dz[np.newaxis,:,:,:]), axis=0)
        self.Db = np.concatenate((Dx2[np.newaxis, :, :, :], Dy2[np.newaxis, :, :, :], Dz2[np.newaxis, :, :, :]), axis=0)
        self.win1.img2 = imageProjection(self.win2.d.img, np.shape(self.win1.d.img),
                                         D = self.D)
        self.win1.contrast2 = self.win2.d.contrast
        self.win1.level2 = self.win2.d.level
        self.win2.img2 = imageProjection(self.win1.d.img, np.shape(self.win2.d.img),
                                         D = self.Db)
        self.win2.contrast2 = self.win1.d.contrast
        self.win2.level2 = self.win1.d.level


if __name__ == "__main__":
    with open('Project4.json') as f:
        dt = json.load(f)
    ctwarp = np.array(dt['ctwarp']['data'])
    ct = np.array(dt['ct']['data'])
    ctvoxsz = np.array(dt['ct']['voxsz'])
    t1 = np.array(dt['t1']['data'])
    t1voxsz = np.array(dt['t1']['voxsz'])

    ipr = interactive_pointBasedRegistration(ct,ctvoxsz,t1,t1voxsz)
    del ipr

    ipr2 = interactive_pointBasedRegistration(ctwarp,ctvoxsz,ct,ctvoxsz)
    del ipr2