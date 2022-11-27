import os
import pickle
import numpy as np
import cv2
import sympy as sym
import json

class Face:
    def __init__(self):
        self.obj_points_large_set = {
            'male':np.array([[20,50,-35],[38,60,-35],[55,55,-40],[38,47,-35],[-20,50,-35],[-38,60,-35],[-55,55,-40],[-38,47,-35],[0,0,0],[0,63,-30]], dtype=np.float64),
            'female':np.array([[17,40,-35],[33,50,-35],[48,45,-35],[33,38,-35], [-17,40,-35],[-33,50,-35],[-48,45,-35],[-33,38,-35], [0,0,0],[0,53,-25]], dtype=np.float64),
            'boy': np.array([[17,40,-35],[33,50,-35],[48,45,-35],[33,38,-35], [-17,40,-35],[-33,50,-35],[-48,45,-35],[-33,38,-35], [0,0,0],[0,53,-25]], dtype=np.float64),
            'girl':np.array([[17,40,-35],[33,50,-35],[48,45,-35],[33,38,-35], [-17,40,-35],[-33,50,-35],[-48,45,-35],[-33,38,-35], [0,0,0],[0,53,-25]], dtype=np.float64)
        }
        self.landmark_indices = [42,43,45,47,39,38,36,40,30,27]#[43,44,46,48,40,39,37,41,31,28]#[362,386,263,374,133,159,33,145,94,168]

    def get_landmark_indices(self):
        return self.landmark_indices

    def get_object_points(self, sex, age):
        key = self.__getkey__(sex.lower(), age)
        if key is not None:
            obj_pts = self.obj_points_large_set[key]
        return obj_pts
    
    def __getkey__(self,sex,age):
        key = None
        if sex=='male' and age>18:
            key = 'male'
        elif sex=='male' and age <= 18:
            key='boy'
        elif sex=='female' and age >18 :
            key = 'female'
        elif sex=='female' and age <= 18:
            key='girl'
        else:
            pass
        return key

class Pose:
    def __init__(self):
        pass

    def get_pose(self):
        raise NotImplementedError

    def compute_covariance(self):
        raise NotImplementedError
    
    def get_covariance(self):
        raise NotImplementedError
        
    def get_rotmat_from_zyx(self, z,y,x, np=False):

        Rz = self.get_rotmat_from_z(z, np)
        Ry = self.get_rotmat_from_y(y, np)
        Rx = self.get_rotmat_from_x(x, np)
        if np:
            R = Rz@Ry@Rx
        else:
            R = Rz*Ry*Rx
        return R


    def get_rotmat_from_x(self, x, use_np=False):
        if use_np:
            x=np.deg2rad(x)
            cosx = np.cos(x)
            sinx = np.sin(x)
            Rx = np.array([[1,0,0],[0,cosx, -sinx], [0, sinx, cosx]])
        else:
            x = sym.rad(x)
            cosx = sym.cos(x)
            sinx = sym.sin(x)
            Rx = sym.Matrix([[1,0,0],[0,cosx, -sinx], [0, sinx, cosx]])
        return Rx

    def get_rotmat_from_y(self, y, use_np=False):
        if use_np:
            y=np.deg2rad(y)
            cosy = np.cos(y)
            siny = np.sin(y)
            Ry = np.array([[cosy, 0, siny],[0, 1 ,0], [-siny, 0, cosy]])
        else:
            y = sym.rad(y)
            cosy = sym.cos(y)
            siny = sym.sin(y)
            Ry = sym.Matrix([[cosy, 0, siny],[0, 1 ,0], [-siny, 0, cosy]])

        return Ry

    def get_rotmat_from_z(self, z, use_np=False):
        if use_np:
            z=np.deg2rad(z)
            cosz = np.cos(z)
            sinz = np.sin(z)
            Rz = np.array([[cosz, -sinz,0],[sinz, cosz, 0], [0,0,1]])
        else:
            z = sym.rad(z)
            cosz = sym.cos(z)
            sinz = sym.sin(z)
            Rz = sym.Matrix([[cosz, -sinz,0],[sinz, cosz, 0], [0,0,1]])
        return Rz

    def decompose_rotmat_to_zyx(self, R:np.ndarray, degrees:bool = True):
        sy=np.sqrt(R[0,0]**2+R[1,0]**2)
        singular = sy <= 1e-6
        if not singular:
            x = np.arctan2(R[2,1],R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2] , R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
        return np.rad2deg(np.array([x,y,z])) if degrees else np.array([x,y,z])

    def pose_to_transform_matrix(self, X:np.ndarray):
        """
        """
        H = np.eye(4)
        R = self.get_rotmat_from_zyx(*X[2::-1])
        RT = -np.dot(R, X[3:])
        H[:3,:3]=R
        H[:3,3]=RT
        return H
    
    def transform_matrix_to_pose(self, H:np.ndarray):
        R=H[:3,:3]
        tvec = H[:3,3]
        T=-np.linalg.inv(R)@tvec
        angles = self.decompose_rotmat_to_zyx(R)
        pose =  np.r_[angles, T.reshape(-1)]
        return pose



class Camera(Pose):
    def __init__(self, calib_file:str=None, input_size:list=None, pickle_path:str='pickles/camera_jacobian.dat'):
        """
        This class represent camera pose with regard to face.
        Args:
            k: [fx, 0 , cx, 0, fy, cy, 0, 0, 1] camera intrinsic matrix.
            dist: [k1, k2, p1, p2, 0] distortion parameter.
        """
        
        with open(calib_file) as f:
            calib = json.load(f)

        self.k=np.array(calib['k'])
        w_ratio = input_size[0]/calib['img_width']
        h_ratio = input_size[1]/calib['img_height']
        self.k[0,0] *= w_ratio
        self.k[0,2] *= w_ratio
        self.k[1,1] *= h_ratio
        self.k[1,2] *= h_ratio
        self.distCoeff = np.array(calib['distCoeff'])
        cx, cy, cz  = sym.symbols('cx cy cz') # camera rotaion r.w.t face
        ctx, cty, ctz = sym.symbols('ctx cty ctz') # camera translation r.w.t face
        symbols = (cx, cy, cz, ctx, cty, ctz)
        x,y,z = sym.symbols('x y z')
        qsymbols=(x,y,z)
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                dpdx = pickle.load(f)
        else:
            dist = sym.Matrix(self.distCoeff)
            K = sym.Matrix(self.k).reshape(3,3)
            PoseVec=sym.Matrix(symbols)
            q = sym.Matrix([x,y,z,1])
            # define H=[R|T] transformation matrix from face to camera point rotaion, P_camera=H*P_face
            R_f2c = self.get_rotmat_from_zyx(cz, cy, cx)
            T_f2c = sym.Matrix([ctx, cty, ctz])
            rT_f2c = -R_f2c*T_f2c
            H_f2c = sym.eye(3,4)
            H_f2c[:3,:3]=R_f2c
            H_f2c[:3,3]=rT_f2c
            pcam = H_f2c*q
            pcam/=pcam[2]
            r2 = pcam[0]**2 + pcam[1]**2
            r4 = r2*r2
            pdist = sym.ones(3,1)
            pdist[0] = pcam[0]*(1+dist[0]*r2+dist[1]*r4)+2*dist[2]*pcam[0]*pcam[1]+dist[3]*(r2+2*pcam[0]**2)
            pdist[1] = pcam[1]*(1+dist[0]*r2+dist[1]*r4)+dist[2]*(r2+2*pcam[1]**2)+2*dist[3]*pcam[0]*pcam[1]
            pimg = K*pdist
            dpdx = pimg[:2,:].jacobian(PoseVec)
            with open('pickles/camera_jacobian.dat', 'wb') as f:
                pickle.dump(dpdx, f)
        
        array2mat = [{"ImmutableDenseMatrix":np.array}, 'numpy']
        self.get_jacobian = sym.lambdify(qsymbols+symbols, dpdx, modules=array2mat)
        self.cam_pose_in_face = None
        self.Cx = None
    
    def set_camera_param(self, fpath:str, width:int, height:int):
        """
            width: detection model input image width
            height: detection model input image height
        """
        with open(fpath) as f:
            calib = json.load(f)

        self.k=np.array(calib['k'])
        w_ratio = width/calib['img_width']
        h_ratio = height/calib['img_height']
        self.k[0,0] *= w_ratio
        self.k[0,2] *= w_ratio
        self.k[1,1] *= h_ratio
        self.k[1,2] *= h_ratio
        self.distCoeff = np.array(calib['distCoeff'])

    def get_reprojection_error_variance(self,obj_pts, landmarks, rvec, tvec):
        img_pts, j = cv2.projectPoints(obj_pts, rvec, tvec, self.k, self.distCoeff)
        img_pts = img_pts.squeeze()
        err = img_pts - landmarks
        n, _ = err.shape
        cov = np.eye(2*n)*np.tile(err.var(0), reps=n)
        return cov


    def project_points(self, obj_points, rvec, tvec, img=None):
         imgpts, jac = cv2.projectPoints(obj_points, rvec, tvec, self.k, self.distCoeff)
         if img is not None:
            for p in imgpts.squeeze():
                cv2.circle(img, (int(p[0]), int(p[1])), 1, [0,255,0], -1)
                
    def estimate_pose_and_cov(self, obj_pts, img_pts, img=None):
        #estimate pose
        # ret, rvec, tvec, inliers = cv2.solvePnPRansac(obj_pts, img_pts.astype(np.float64), self.k, self.distCoeff, flags = cv2.SOLVEPNP_ITERATIVE,
        # confidence=0.9999, reprojectionError=3, iterationsCount=100)#cv2.SOLVEPNP_EPNP
        ret, rvec, tvec = cv2.solvePnP(obj_pts, img_pts.astype(np.float64), self.k, self.distCoeff, flags = cv2.SOLVEPNP_EPNP)
        if ret:
            # imgpts, jac = cv2.projectPoints(obj_pts, rvec, tvec, self.k, self.distCoeff)
            # project = np.sqrt(((imgpts.squeeze()-img_pts)**2).sum(axis=1))
            # print(project)
            if img is not None:
                self.project_points(obj_pts, rvec, tvec, img)
                cv2.imwrite('gaze.png', img)
            Cp = self.get_reprojection_error_variance(obj_pts, img_pts, rvec, tvec)
            R = cv2.Rodrigues(rvec)
            angle = self.decompose_rotmat_to_zyx(R[0])
            T=-np.dot(np.linalg.inv(R[0]), tvec)
            self.cam_pose_in_face=np.r_[angle,T.reshape(-1)]

            M=[]
            for p in obj_pts:
                M.append(self.get_jacobian(*p,*self.cam_pose_in_face))
            M = np.vstack(M)
            J = np.linalg.inv((M.T@M))@M.T
            self.Cx = J@Cp@J.T
        else:
            self.cam_pose_in_face = None
            self.Cx = None 
        return self.cam_pose_in_face, self.Cx

    def get_pose(self):
        return self.cam_pose_in_face

    def get_cov(self):
        self.Cx

class World(Pose):
    def __init__(self, calib_file, dzdx_path='pickles/world_dZdX.dat', dzdw_path='pickles/world_dZdW.dat'):

        with open(calib_file) as f:
            calib = json.load(f)
        self.rvec = np.array(calib['rvec'])
        self.tvec = np.array(calib['tvec'])
        R = cv2.Rodrigues(self.rvec)[0]
        R = np.linalg.inv(R)
        angle = self.decompose_rotmat_to_zyx(R)
        self.world_pose_in_cam = np.r_[angle,self.tvec.reshape(-1) ]
        cx, cy, cz = sym.symbols('cx cy cz')
        ctx, cty ,ctz = sym.symbols('ctx cty ctz')
        cam_pose_syms = (cx, cy, cz, ctx, cty, ctz)
        wx, wy,wz = sym.symbols('wx wy wz')
        wtx, wty, wtz = sym.symbols('wtx wty wtz')
        world_pose_syms = (wx, wy, wz, wtx, wty, wtz)
        if os.path.exists(dzdx_path) and os.path.exists(dzdw_path):
            with open(dzdx_path, 'rb') as f:
                dZdX = pickle.load(f)
            with open(dzdw_path, 'rb') as f:
                dZdW = pickle.load(f)
        else:

            #let call Camera pose as X
            X = sym.Matrix(cam_pose_syms)
            #defile camera pose H
            R_f2c = self.get_rotmat_from_zyx(cz, cy, cx)
            rT_f2c = -R_f2c*sym.Matrix([ctx, cty, ctz])
            H_f2c = sym.eye(4)
            H_f2c[:3,:3]=R_f2c
            H_f2c[:3,3] = rT_f2c

            #define camera to world transform
        
            #let call world pose as W
            W= sym.Matrix(world_pose_syms)
            R_c2w = self.get_rotmat_from_zyx(wz, wy, wx)
            rT_c2w = -R_c2w*sym.Matrix([wtx, wty, wtz])
            H_c2w = sym.eye(4)
            H_c2w[:3,:3] = R_c2w
            H_c2w[:3,3] = rT_c2w

            H_f2w = H_c2w*H_f2c
            H_f2w = H_f2w[:3,:]
            R_f2w = H_f2w[:,:3].T #_rep_to_field().inv().to_Matrix()

            #define world pose with regard to face
            Z = sym.zeros(6,1)
            sy = sym.sqrt(R_f2w[0,0]**2+R_f2w[1,0]**2)
            Z[0] = sym.atan2(R_f2w[2,1], R_f2w[2,2])
            Z[1] = sym.atan2(-R_f2w[2,0], sy)
            Z[2] = sym.atan2(R_f2w[2,1], R_f2w[2,2])
            Z[3:,:] = H_f2w[:,3]
            dZdX = Z.jacobian(X)
            dZdW = Z.jacobian(W)
            with open(dzdx_path, 'wb') as f:
                    pickle.dump(dZdX, f)

            with open(dzdw_path, 'wb') as f:
                    pickle.dump(dZdW, f)

        array2mat = [{'ImmutableDenseMatrix': np.array}, 'numpy']
        self.get_dZdX=sym.lambdify(cam_pose_syms+world_pose_syms, dZdX, modules=array2mat)
        self.get_dZdW=sym.lambdify(cam_pose_syms+world_pose_syms, dZdW, modules=array2mat)

    def set_camera_param(self, fpath):
        with open(fpath) as f:
            calib = json.load(f)
        self.rvec = np.array(calib['rvec'])
        self.tvec = np.array(calib['tvec'])   
    
    def jacobian(self, X:list):
        Jx = self.get_dZdX(*X.tolist(), *self.world_pose_in_cam.tolist())
        Jw = self.get_dZdW(*X.tolist(), *self.world_pose_in_cam.tolist())
        return Jx, Jw

    def get_pose(self):
        return self.world_pose_in_cam


class Gaze(Pose):
    def __init__(self, calib_file:str, input_size:list, pickle_path:str='pickles/camera_jacobian.dat',
     dzdx_path:str='pickles/world_dZdX.dat', dzdw_path:str='pickles/world_dZdW.dat'):
        """
        args:
        calib_file: path to camera calibration information file
        input_size: input image size (width,height) 
        """
        self.camera = Camera(calib_file, input_size, pickle_path)
        self.world = World(calib_file, dzdx_path, dzdw_path)
        x,y,z = sym.symbols('x y z')
        tx, ty, tz = sym.symbols('tx ty tz')

        syms = (x, y, z, tx, ty, tz)
        X = sym.Matrix(syms)
        
        R_f2w = self.get_rotmat_from_zyx(x,y,z)
        T_f2w = sym.Matrix([tx,ty,tz])
        dirvec = R_f2w[:2]
        t = -T_f2w[1]/R_f2w[1,1]
        gazePoint = T_f2w+t*R_f2w[:,2]
        dgdx = gazePoint.jacobian(X)


        array2mat = [{'ImmutableDenseMatrix': np.array}, 'numpy']
        self.func_jacobian=sym.lambdify(syms, dgdx, modules=array2mat)

    def set_calib_param(self, fpath:str, width:int, height:int)->None:
        self.camera.set_camera_param(fpath, width, height)
        self.world.set_camera_param(fpath)
        
    def jacobian(self, X):
        J = self.func_jacobian(*X)
        return J

    def estimate_gaze_point(self, obj_pts, img_pts, img=None):
        """
        """
        # X : state of camera, CovX: covariance matrix of camera
        X, CovX = self.camera.estimate_pose_and_cov(obj_pts, img_pts, img)
        if X is not None and CovX is not None:
            Jx, Jw = self.world.jacobian(X)
            Cy = Jx@CovX@Jx.T

            # transformation matrix computation
            Hf2c=self.camera.pose_to_transform_matrix(X)
            Hc2w = self.world.pose_to_transform_matrix(self.world.get_pose())
            Hf2w=Hc2w@Hf2c
            X_f2w=self.transform_matrix_to_pose(Hf2w)
            J = self.jacobian(X_f2w)
            C_gaze=J@Cy@J.T
            #gaze point = 
            p_face = Hf2w[:3,3]
            v_gaze = Hf2w[:3,2]
            t=-p_face[1]/v_gaze[1]
            gaze_point = p_face+v_gaze*t
            #self.get_error_ellipse(gaze_point.astype(int), C_gaze)
            return gaze_point[::2], C_gaze[::2,::2]
        return None, None
    
    def get_error_ellipse(self, gaze, Cov):
        C = Cov[::2,::2]
        eigval, eigvec = np.linalg.eig(C)
        major_val, major_axis = eigval[1], eigvec[:,1]
        minor_val, minor_axis = eigval[0], eigvec[:,0]
        h_mj_len = np.sqrt(5.991*major_val) # 95% probability error
        h_mn_len = np.sqrt(5.991*minor_val)
        ort = np.rad2deg(np.arctan2(major_axis[1], major_axis[0]))
        canvas=np.zeros(np.r_[gaze[::-1]*2,3], dtype=np.uint8)
        
        ellipse_float = ((gaze[0], gaze[1]), (h_mj_len, h_mn_len), ort)
        canvas = cv2.ellipse(canvas, ellipse_float, [0,255,0],2)
        canvas = cv2.circle(canvas, (gaze[0], gaze[1]), 3, [0,0,255],-1)
        cv2.imwrite('error.png', canvas)








