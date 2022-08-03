import autograd.numpy as anp
from autograd import jacobian
class Jacobian:
    def run(self, state):
        R = self.get_rotmat_from_zyx(*state[3:])

        a = jacobian(self.get_rotmat_from_zyx,(0,1,2))
        print(R)
        res = a(*state[3:])
        print(res)

    def get_rotmat_from_zyx(self, x,y,z):
        Rz = self.get_rotmat_from_z(z)
        Ry = self.get_rotmat_from_y(y)
        Rx = self.get_rotmat_from_x(x)
        R = anp.dot(Rz, anp.dot(Ry,Rx))
        return R
        

    def get_rotmat_from_x(self, x):
        cosx = anp.cos(x)
        sinx = anp.sin(x)
        Rx = anp.array([[1,0,0],[0,cosx, -sinx], [0, sinx, cosx]])
        return Rx

    def get_rotmat_from_y(self, y):
        cosy = anp.cos(y)
        siny = anp.sin(y)
        Ry = anp.array([[cosy, 0, siny],[0, 1 ,0], [-siny, 0, cosy]])
        return Ry

    def get_rotmat_from_z(self, z):
        cosz = anp.cos(z)
        sinz = anp.sin(z)
        Rz = anp.array([[cosz, -sinz,0],[sinz, cosz, 0], [0,0,1]])
        return Rz

if __name__ =="__main__":
    j = Jacobian()
    j.run(anp.array([1,2,3,4,5,6],dtype=anp.float32))