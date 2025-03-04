import numpy as np
from scipy.optimize import linear_sum_assignment
import os.path
import configparser
from collections import deque
### Module imports ###
import sys
import warnings

warnings.filterwarnings("ignore")
from tracking.matcher.Camera import Camera
import torch
from tracking.matcher.Triangulate import Triangulate


class TrackletMatcher:
    """
    Class implementation for associating 2D tracklets into 3D tracklets        
    """

    def __init__(self, dataPath):
        """
        Initialize object
        
        Input:
            dataPath: String path to the main folder
        """

        # Load settings and data
        self.loadSettings(dataPath)

        self.cams = self.prepareCams(dataPath)  # Load camera objects

        # Internal stuff
        self.triangulated = {}

    def loadSettings(self, path):
        """
        Load settings from config file in the provided path.
        
        Config file includes information on the following, which is set in the object:
            reprojection_err_mean: The mean value of a Gaussian distribution of reprojection errors
            reprojection_err_std: The standard deviation of a Gaussian distribution of reprojection errors
            movement_err_mean: The mean value of a Gaussian distribution of movement errors
            movement_err_std: The standard deviation of a Gaussian distribution of movement errors
            same_view_max_overlap: The maximum allowed frame overlap of two tracklets
            tracklet_min_length: Minimum trackelt length
            camera_1_sync_frame: Sync frame for camera 1
            camera_2_sync_frame: Sync frame for camera 2 
            
        Input:
            path: String path to the folder where the settings.ini file is located
        """

        config = self.readConfig(path)

        # Get tracklet matching parameters
        c = config['TrackletMatcher']
        self.reprojMeanErr = c.getfloat('reprojection_err_mean')
        self.reprojStdErr = c.getfloat('reprojection_err_std')
        self.movErrMean = c.getfloat('movement_err_mean')
        self.movErrStd = c.getfloat('movement_err_std')
        self.sameViewMaxOverlap = c.getint('same_view_max_overlap')
        self.trackletMinLength = c.getint('tracklet_min_length')
        self.temporalPenalty = c.getint('temporal_penalty')
        self.FPS = c.getint('FPS')
        self.camera2_useHead = c.getboolean("cam2_head_detector", False)

        # Get aquarium size
        c = config['Aquarium']
        self.maxX = c.getfloat("aquarium_width")
        self.maxY = c.getfloat("aquarium_depth")
        self.maxZ = c.getfloat("aquarium_height", np.inf)

        self.minX = c.getfloat("min_aquarium_width", 0.0)
        self.minY = c.getfloat("min_aquarium_depth", 0.0)
        self.minZ = c.getfloat("min_aquarium_height", 0.0)

        print("Aquarium Dimensions\n\tX: {} - {}\n\tY: {} - {}\n\tZ: {} - {}\n".format(self.minX, self.maxX, self.minY,
                                                                                       self.maxY, self.minZ, self.maxZ))
        # Get camera synchronization parameters
        c = config['CameraSynchronization']
        cam1frame = c.getint('cam1_sync_frame')
        cam2frame = c.getint('cam2_sync_frame')
        self.camera1_offset = max(0, cam2frame - cam1frame)
        self.camera2_offset = max(0, cam1frame - cam2frame)
        self.camera1_length = c.getint("cam1_length")
        self.camera2_length = c.getint("cam2_length")

    def withinAquarium(self, x, y, z):
        """
        Checks whether the provided x,y,z coordinates are inside the aquarium.
        
        Input:
            x: x coordinate
            y: y coordinate
            z: z coordinate
        
        Output:
            Boolean value stating whether the point is inside the aquarium
        """

        if (x < self.minX or x > self.maxX):
            return False
        if (y < self.minY or y > self.maxY):
            return False
        if (z < self.minZ or z > self.maxZ):
            return False
        return True

    def prepareCams(self, path):
        """
        Loads the camera objects stored in a pickle file at the provided path

        Input:
            path: Path to the folder where the camera.pkl file is located

        Output:
            cams: A dict containing the extracted camera objects
        """

        cam1Path = os.path.join(path, 'cam1_intrinsic.json')
        cam2Path = os.path.join(path, 'cam1_intrinsic.json')
        if (not os.path.isfile(cam1Path)):
            print("Error finding camera calibration file: \n {0}".format(cam1Path))
            sys.exit(0)
        if (not os.path.isfile(cam2Path)):
            print("Error finding camera calibration file: \n {0}".format(cam2Path))
            sys.exit(0)

        cam1ref = os.path.join(path, 'cam1_references.json')
        cam2ref = os.path.join(path, 'cam2_references.json')
        if (not os.path.isfile(cam1ref)):
            print("Error finding camera corner reference file: \n {0}".format(cam1ref))
            sys.exit(0)
        if (not os.path.isfile(cam2ref)):
            print("Error finding camera corner reference file: \n {0}".format(cam2ref))
            sys.exit(0)

        cams = {}
        # cam1 = joblib.load(cam1Path)
        cam1 = Camera(cam1Path, cam1ref)
        cam1.calcExtrinsicFromJson(cam1ref)
        cams[1] = cam1

        # cam2 = joblib.load(cam2Path)
        cam2 = Camera(cam2Path, cam2ref)
        cam2.calcExtrinsicFromJson(cam2ref)
        cams[2] = cam2

        print("")
        print("Camera 1:")
        print(" - position: \n" + str(cam1.getPosition()))
        print(" - rotation: \n" + str(cam1.getRotationMat()))
        print("")

        print("Camera 2:")
        print(" - position: \n" + str(cam2.getPosition()))
        print(" - rotation: \n" + str(cam2.getRotationMat()))
        print("")

        return cams

    def readConfig(self, path):
        """
        Reads the settings.ini file located at the specified directory path
        If no file is found the system is exited

        Input:
            path: String path to the directory

        Output:
            config: A directory of dicts containing the configuration settings
        """

        config = configparser.ConfigParser(inline_comment_prefixes='#')
        configFile = os.path.join(path, 'settings.ini')
        if (os.path.isfile(configFile)):
            config.read(configFile)
            return config
        else:
            print("Error loading configuration file:\n{0}\n Exiting....".format(configFile))
            sys.exit(0)

    def get_3d_tracks(self, tracks1, tracks2):
        track_pairs = self.associate_tracks(tracks1, tracks2)
        tracks_3d = []
        if track_pairs is not None and len(track_pairs) != 0:
            for top, front in track_pairs.items():
                track_3d_dict = dict()
                # ToDo: Is this right?
                track_3d_dict['id'] = front.id
                top_pos = top.pos[0].cpu().numpy()
                front_pos = front.pos[0].cpu().numpy()
                points, distance, error = self.get_3d_pos_by_points(top_pos, front_pos)
                track_3d_dict['err'] = error
                track_3d_dict['3d_x'] = points[0]
                track_3d_dict['3d_y'] = points[1]
                track_3d_dict['3d_z'] = points[2]
                # center points
                track_3d_dict['cam1_x'] = (top_pos[0] + top_pos[2]) / 2
                track_3d_dict['cam1_y'] = (top_pos[1] + top_pos[3]) / 2
                track_3d_dict['cam2_x'] = (front_pos[0] + front_pos[2]) / 2
                track_3d_dict['cam2_y'] = (front_pos[1] + front_pos[3]) / 2
                track_3d_dict['cam1_tl_x'] = top_pos[0]
                track_3d_dict['cam1_tl_y'] = top_pos[1]
                track_3d_dict['cam1_c_x'] = (top_pos[0] + top_pos[2]) / 2
                track_3d_dict['cam1_c_y'] = (top_pos[1] + top_pos[3]) / 2
                track_3d_dict['cam1_w'] = top_pos[2] - top_pos[0]
                track_3d_dict['cam1_h'] = top_pos[3] - top_pos[1]
                track_3d_dict['cam2_tl_x'] = front_pos[0]
                track_3d_dict['cam2_tl_y'] = front_pos[1]
                track_3d_dict['cam2_c_x'] = (front_pos[0] + front_pos[2]) / 2
                track_3d_dict['cam2_c_y'] = (front_pos[1] + front_pos[3]) / 2
                track_3d_dict['cam2_w'] = front_pos[2] - front_pos[0]
                track_3d_dict['cam2_h'] = front_pos[3] - front_pos[1]
                tracks_3d.append(track_3d_dict)
        return tracks_3d

    def associate_tracks(self, tracks1, tracks2):
        # Todo: process the vocation
        if not tracks1 or not tracks2:
            return []
        if len(tracks1) == len(tracks2):
            return self.associate_tracks_by_x_order(tracks1, tracks2)
        else:
            return self.associate_tracks_by_horizontal_distance(tracks1, tracks2)

    def associate_tracks_by_x_order(self, tracks1, tracks2):
        sorted_tracks1 = sorted(tracks1, key=lambda t: t.pos[0][0])
        sorted_tracks2 = sorted(tracks2, key=lambda t: t.pos[0][0])

        if len(sorted_tracks1) != len(sorted_tracks2):
            raise ValueError("The number of tracks in both sets must be the same.")

        associations = {}

        for i in range(len(sorted_tracks1)):
            associations[sorted_tracks1[i]] = sorted_tracks2[i]

        return associations

    def associate_tracks_by_horizontal_distance(self, tracks1, tracks2):
        # Create a distance matrix to store distances between tracks1 and tracks2
        distances = np.zeros((len(tracks1), len(tracks2)))

        # Calculate distances between all pairs of tracks
        for i, track1 in enumerate(tracks1):
            for j, track2 in enumerate(tracks2):
                distances[i, j] = abs(track1.pos[0][0] - track2.pos[0][0])

        # If all distances are infinity, no associations can be made
        if np.all(np.isinf(distances)):
            return {}

        # Initial associations based on minimum distances
        row_indices, col_indices = linear_sum_assignment(distances)
        associations = {tracks1[i]: tracks2[j] for i, j in zip(row_indices, col_indices)}

        return associations

    def get_3d_pos_by_points(self, top, front):
        tr = Triangulate()
        top = (top[0] + top[2]) / 2, (top[1] + top[3]) / 2
        front = (front[0] + front[2]) / 2, (front[1] + front[3]) / 2
        p, d = tr.triangulatePoint(top, front, self.cams[1], self.cams[2], correctRefraction=True)

        p1 = self.cams[1].forwardprojectPoint(*p)
        p2 = self.cams[2].forwardprojectPoint(*p)

        pos1 = np.array(top)
        err1 = np.linalg.norm(pos1 - p1)
        pos2 = np.array(front)
        err2 = np.linalg.norm(pos2 - p2)
        err = err1 + err2
        return p, d, err


class Track_3d(object):
    def __init__(self, pos, track_id, inactive_patience):
        self.id = track_id
        self.pos = pos
        self.inactive_patience = inactive_patience
        self.last_pos = deque([pos.clone()])
        self.last_v = torch.Tensor([])

    def reset_last_pos(self):
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())
