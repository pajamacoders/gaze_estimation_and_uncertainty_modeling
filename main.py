import numpy as np
import sympy as sym
sym.init_printing()
class TransformMatrix:
    def __init__(self): 
        self.x, self.y, self.z  = sym.symbols('x y z')
        self.tx, self.ty, self.tz = sym.symbols('tx ty tz')
        # define H=[R|T] transformation matrix from face to camera point rotaion, P_camera=H*P_face
        self.R_f2c = self.get_rotmat_from_zyx()
        self.T_f2c = sym.Matrix([self.tx, self.ty, self.tz])
        self.rT_f2c = -self.R_f2c*self.T_f2c
        self.H_f2c = sym.eye(4)
        self.H_f2c[:3,:3]=self.R_f2c
        self.H_f2c[:3,3]=self.rT_f2c

        # define H=[R|T] transformation matrix from camera to world point rotaion, P_world=H*P_camera
        self.R_c2w = self.get_rotmat_from_zyx()
        self.T_c2w = sym.Matrix([self.tx, self.ty, self.tz])
        self.rT_c2w = -self.R_c2w*self.T_c2w
        self.H_c2w = sym.eye(4)
        self.H_c2w[:3,:3]=self.R_c2w
        self.H_c2w[:3,3]=self.rT_c2w

        self.H_f2w=self.H_c2w*self.H_f2c
        self.H_f2w=self.H_f2w[:3,:]
        self.R_f2w = self.H_f2w[:3,:3]

    def decompose2state(self, H,R):
        pass

        
    def run(self, x,y,z,tx,ty,tz):
        values={'x':x, 'y':y, 'z':np.deg2rad(z)}
        R = self.get_rotmat_from_zyx()
        T = sym.Matrix([self.tx, self.ty, self.tz])
        rT = -R*T
        R
        R = R.reshape(9,1)
        X = sym.Matrix([self.x, self.y, self.z])
        res = R.jacobian(X)
        print(res)

    def get_rotmat_from_zyx(self):
        Rz = self.get_rotmat_from_z(self.z)
        Ry = self.get_rotmat_from_y(self.y)
        Rx = self.get_rotmat_from_x(self.x)
        R = Rz*Ry*Rx
        return R


    def get_rotmat_from_x(self, x):
        cosx = sym.cos(x)
        sinx = sym.sin(x)
        Rx = sym.Matrix([[1,0,0],[0,cosx, -sinx], [0, sinx, cosx]])
        return Rx

    def get_rotmat_from_y(self, y):
        cosy = sym.cos(y)
        siny = sym.sin(y)
        Ry = sym.Matrix([[cosy, 0, siny],[0, 1 ,0], [-siny, 0, cosy]])
        return Ry

    def get_rotmat_from_z(self, z):
        cosz = sym.cos(z)
        sinz = sym.sin(z)
        Rz = sym.Matrix([[cosz, -sinz,0],[sinz, cosz, 0], [0,0,1]])
        return Rz

if __name__=="__main__":
    j = TransformMatrix()
    j.run(0,0,90)