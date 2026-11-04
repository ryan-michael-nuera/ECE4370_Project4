import json
import numpy as np
from volumeViewer import *
import time
from Probe_to_Image import *
import copy

class DisplayReg(volumeViewer):
    def __init__(self):
        super().__init__()
        self.renwin2 = vtk.vtkRenderWindow()
        self.ren2 = vtk.vtkRenderer()
        self.renwin2.AddRenderer(self.ren2)
        self.renwin2.Render()

    def update(self,direction=-1,slc=-1,level=float("nan"),contrast=float("nan"),resize=-1):
        super().update(direction,slc,level,contrast,resize)
        if direction == 3:
            self.ResetCameraClippingRange()
            self.renwin.Render()
            self.inter.ProcessEvents()
            self.UpdateCam2()

    def UpdateCam2(self):
        self.renwin2.MakeCurrent()
        self.ren2.RemoveAllViewProps()
        res = self.GetActors()
        res.InitTraversal()
        a = res.GetNextActor()
        while a is not None:
            self.ren2.AddActor(a)
            a = res.GetNextActor()

        cam = self.GetActiveCamera()
        cam2 = vtk.vtkCamera()
        cam2.DeepCopy(cam)
        cam2.Azimuth(90)
        self.ren2.SetActiveCamera(cam2)

        self.ren2.ResetCameraClippingRange()
        self.renwin2.Render()

    def Display(self, blocking=True):
        self.update()
        if blocking:
            while (self.quit == False):
                self.ResetCameraClippingRange()
                self.renwin.Render()
                self.inter.ProcessEvents()
                self.UpdateCam2()
    def __del__(self):
        del self.renwin2,self.ren2
        super().__del__()


def project4_driver(trk, probepnt):
    with open('Project4_8.json') as f:
        dt = json.load(f)
    mr = np.array(dt['mr']['data'], dtype=np.int16)
    Tt1_f2 = np.array(dt['Tt1_f2'], dtype=np.float64)
    frame_V = np.array(dt['frame_V'], dtype=np.float64)
    marker_V = np.array(dt['marker_V'], dtype=np.float64)
    pointer_V = np.array(dt['pointer_V'], dtype=np.float64)
    frame_F = np.array(dt['frame_F'], dtype=np.longlong)
    marker_F = np.array(dt['marker_F'], dtype=np.longlong)
    pointer_F = np.array(dt['pointer_F'], dtype=np.longlong)
    targets = np.array(dt['targets'], dtype=np.float64)

    d = DisplayReg()
    d.setImage(mr,np.array(dt['mr']['voxsz']))

    # set camera position?


    frame = surface()
    frame.verts = frame_V
    frame.faces = frame_F
    d.addSurf(frame.verts, frame.faces, color=[0.2, 0.2, 0.2], opacity=1)

    pointer = surface()
    pointer.verts = pointer_V
    pointer.faces = pointer_F
    d.addSurf(pointer.verts, pointer.faces, color=[0.7, 0.7, 0.7], opacity=1)

    markers = surface()
    markers.verts = marker_V
    markers.faces = marker_F
    d.addSurf(markers.verts, markers.faces, color=[.95, .95, .95], opacity=1)

    n = trk.GetNumTools()

    target_frame = copy.copy(frame)
    target_pointer = copy.copy(pointer)
    target_markers = copy.copy(markers)
    d.addSurf(target_frame.verts, target_frame.faces, color=[.2,0,0], opacity=1)
    d.addSurf(target_pointer.verts, target_pointer.faces, color=[.7, 0, 0], opacity=1)
    d.addSurf(target_markers.verts, target_markers.faces, color=[.95, 0, 0], opacity=1)


    lvrs = surface()
    lvrs.verts = np.array(dt['lvr']['Vertices'])
    lvrs.faces = np.array(dt['lvr']['Faces'])
    d.addSurf(lvrs.verts, lvrs.faces, color=[1.0, 0.7, 0.7], opacity=1)
    d.update()

    for i in range(np.shape(targets)[0]):
        targethit = 0
        start = time.time()
        while d.quit==0 and targethit==0:
            Tr = trk.GetTools(n)
            testbedPhi = Tr[:, :, 1]
            pointerPhi = Tr[:, :, 2]
            if testbedPhi[0,3] != 0 and pointerPhi[0,3] != 0:
                x = Probe_to_Image(Tt1_f2, testbedPhi, pointerPhi, probepnt)

            elif testbedPhi[0,3] == 0:
                print('Demo board blocked from tracker!')
                continue
            elif pointerPhi[0,3] == 0:
                print('Pointer tool not visible by tracker!')
                continue

            d.objlist[0].updateActor(Probe_to_Image(Tt1_f2, testbedPhi, pointerPhi, frame_V))
            d.objlist[1].updateActor(Probe_to_Image(Tt1_f2, testbedPhi, pointerPhi, pointer_V))
            d.objlist[2].updateActor(Probe_to_Image(Tt1_f2, testbedPhi, pointerPhi, marker_V))
            d.objlist[3].updateActor(Probe_to_Image(Tt1_f2, testbedPhi, testbedPhi @ targets[i, :, :], frame_V))
            d.objlist[4].updateActor(Probe_to_Image(Tt1_f2, testbedPhi, testbedPhi @ targets[i, :, :], pointer_V))
            d.objlist[5].updateActor(Probe_to_Image(Tt1_f2, testbedPhi, testbedPhi @ targets[i, :, :], marker_V))

            tpnt = Probe_to_Image(Tt1_f2, testbedPhi, testbedPhi @ targets[i, :, :], probepnt)
            targeterror = np.sqrt(np.sum((tpnt[0:3] - x[0:3])**2))
            if targeterror < 10:
                targethit = 1
            else:
                targethit = 0

            print(f'Target error = {targeterror: .1f} mm')
            d.update(direction=3)
            
        t = time.time() - start
        print(f'You found the target in {t:.1f} seconds with target error of ' +\
                 f'{targeterror:.1f} mm. Press enter to continue')
        if d.quit:
            break
        input()

