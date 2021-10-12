import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

from IPython.display import Image

from carrada_utils.utils import CARRADA_HOME
from carrada_utils.utils.configurable import Configurable
from carrada_utils.utils.visualize_signal import SignalVisualizer
from carrada_utils.utils.transform_annotations import AnnotationTransformer
from carrada_utils.utils.generate_annotations import AnnotationGenerator
from carrada_utils.utils.transform_data import DataTransformer

## %run ../carrada_utils/scripts/set_path.py './components/carrada_datasets/'


class RadarLoader:
    def __init__(self, seq_name):
        self.seq_name = seq_name
        self.config_path = os.path.join(CARRADA_HOME, "config.ini")
        self.config = Configurable(self.config_path).config
        self.warehouse = self.config["data"]["warehouse"]
        self.carrada = os.path.join(self.warehouse)
        self.seq_path = os.path.join(self.carrada, self.seq_name)
        annotations_path = os.path.join(self.carrada, "annotations_frame_oriented.json")
        # Load data

        # Annotations
        with open(annotations_path, "r") as fp:
            annotations = json.load(fp)
        self.annotations = annotations[self.seq_name]

    def load_data_from_frame(self, frame_name):
        rd_path = os.path.join(
            self.seq_path, "range_doppler_numpy", frame_name + ".npy"
        )
        ra_path = os.path.join(self.seq_path, "range_angle_numpy", frame_name + ".npy")
        img_path = os.path.join(self.seq_path, "camera_images", frame_name + ".jpg")

        """
        # Path to the RAD tensor (soon available)
        rad_path = os.path.join(seq_path, 'RAD_numpy', frame_name + '.npy')
        """

        annotations_path = os.path.join(self.carrada, "annotations_frame_oriented.json")
        # Load data

        # Annotations
        with open(annotations_path, "r") as fp:
            annotations = json.load(fp)
        self.annotations = annotations[self.seq_name]

        # Range-angle and range-Doppler matrices
        """
        # the RA and RD matrices can be computed directly from the RAD tensor
        rad_matrix = np.load(rad_path)
        data_transformer = DataTransformer(rad_matrix)
        ra_matrix = data_transformer.to_ra()
        rd_matrix = data_transformer.to_rd()
        """
        ra_matrix = np.load(ra_path)
        rd_matrix = np.load(rd_path)
        return ra_matrix, rd_matrix, annotations

    def display_color_image(self, frame_name):
        # Camera image of the scene
        img_path = os.path.join(self.seq_path, "camera_images", frame_name + ".jpg")
        print(img_path)
        print(
            "Camera image of the scene {}, frame {}".format(self.seq_name, frame_name)
        )
        img = cv2.imread(img_path)
        cv2.imshow("image", img)
        cv2.waitKey(0)

    def display_color_image_stream(self, frame_name):
        # Camera image of the scene
        for i in range(100, 1000):
            img_path = os.path.join(
                self.seq_path, "camera_images", "000" + str(i) + ".jpg"
            )
            img = cv2.imread(img_path)
            cv2.imshow("image", img)
            cv2.waitKey(200)

    def get_color_image_datastream(self, resize=(64, 64)):
        data = []
        # Camera image of the scene
        data = []
        file_list = os.listdir(os.path.join(self.seq_path, "camera_images"))
        for file_path in sorted(file_list):
            img_path = os.path.join(self.seq_path, "camera_images", file_path)
            img = cv2.imread(img_path)

            data.append(img)

        return data

    def display_range_doppler(self, frame_name):
        ra_matrix, rd_matrix, annotations = self.load_data_from_frame(frame_name)
        # Range-Doppler visualization
        signal_visualizer = SignalVisualizer(rd_matrix)
        print("Raw Range-Doppler representation:")
        # print(signal_visualizer.image)

        cv2.imshow("image", signal_visualizer.image)
        cv2.waitKey(0)

    def display_range_doppler_stream(self, frame_name):

        print("Raw Range-Doppler representation stream:")
        for i in range(100, 1000):
            rd_path = os.path.join(
                self.seq_path, "range_doppler_numpy", "000" + str(i) + ".npy"
            )
            rd_matrix = np.load(rd_path)
            # Range-Doppler visualization
            signal_visualizer = SignalVisualizer(rd_matrix)
            # print(signal_visualizer.image)

            cv2.imshow("image", signal_visualizer.image)
            cv2.waitKey(500)

    def display_range_angle(self, frame_name):
        ra_matrix, rd_matrix, annotations = self.load_data_from_frame(frame_name)
        # Range-Doppler visualization
        signal_visualizer = SignalVisualizer(ra_matrix)
        print("Raw Range-Angle representation:")
        # print(signal_visualizer.image)

        cv2.imshow("image", signal_visualizer.image)
        cv2.waitKey(0)

    def display_range_angle_stream(self, frame_name):

        print("Raw Range-Doppler representation stream:")
        for i in range(100, 999):
            ra_path = os.path.join(
                self.seq_path, "range_angle_numpy", "000" + str(i) + ".npy"
            )
            ra_path_2 = os.path.join(
                self.seq_path, "range_angle_numpy", "000" + str(i + 1) + ".npy"
            )
            ra_matrix = np.load(ra_path)
            ra_matrix_2 = np.load(ra_path_2)
            diff_ra_matrix = ra_matrix_2 - ra_matrix
            # Range-Doppler visualization
            signal_visualizer = SignalVisualizer(diff_ra_matrix)
            # print(signal_visualizer.image)
            resized = cv2.resize(signal_visualizer.image, (64, 64))
            cv2.imshow("image", resized)
            cv2.waitKey(500)

    def get_range_angle_stream_data(
        self, clip_and_normalize=False, resize=(64, 64), mean_normalization=False
    ):
        data = []
        file_list = os.listdir(os.path.join(self.seq_path, "range_angle_numpy"))
        ra_path = os.path.join(self.seq_path, "range_angle_numpy", "000000.npy")
        ra_matrix = np.load(ra_path)
        size_bf = ra_matrix.shape
        for file_path in sorted(file_list):
            ra_path = os.path.join(self.seq_path, "range_angle_numpy", file_path)
            ra_matrix = np.load(ra_path)
            if clip_and_normalize:
                entry_ = cv2.resize(
                    ra_matrix.clip(0)
                    / np.max(
                        ra_matrix,
                    ),
                    resize,
                )
                data.append(entry_)
            else:
                data.append(ra_matrix)
        return data, size_bf

    def get_range_angle_stream_data_differentiated(
        self, clip_and_normalize=False, resize=(64, 64)
    ):
        data = []
        file_list = sorted(os.listdir(os.path.join(self.seq_path, "range_angle_numpy")))
        file_before = None
        for file_path in sorted(file_list):
            if file_before is None:
                file_before = file_path
                continue
            ra_path = os.path.join(self.seq_path, "range_angle_numpy", file_path)
            ra_path_bf = os.path.join(self.seq_path, "range_angle_numpy", file_before)
            file_before = file_path
            ra_matrix = np.load(ra_path)
            ra_matrix_bf = np.load(ra_path_bf)

            if clip_and_normalize:
                clipped_normalized = ra_matrix / np.max(
                    ra_matrix
                ) - ra_matrix_bf / np.max(ra_matrix_bf)
                data.append(
                    cv2.resize(
                        clipped_normalized.clip(0)
                        / np.max(
                            clipped_normalized,
                        ),
                        resize,
                    )
                )
            else:
                data.append(ra_matrix - ra_matrix_bf)
        return data

    def get_range_doppler_stream_data(self):
        data = []
        file_list = os.listdir(os.path.join(self.seq_path, "range_doppler_numpy"))
        for file_path in file_list:
            rd_path = os.path.join(self.seq_path, "range_doppler_numpy", file_path)
            rd_matrix = np.load(rd_path)
            data.append(rd_matrix)
        return data

    def get_annotations(self):
        return self.annotations

    def visualize_annotations(self, differentiated=False, size_bf=(64, 64)):
        annotations = self.get_annotations()
        print("HERE")
        dense_ = []
        dense_mp_visualization = []
        for entry in annotations:
            arrr = np.zeros(size_bf)
            arrr_mp_dense = np.zeros(size_bf)
            for ittem in annotations[entry]:
                points = annotations[entry][ittem]["range_angle"]["dense"]
                annot_pts = np.array(points)
                mean_point = np.sum(np.array(points), axis=0) // len(np.array(points))
                arrr_mp_dense[mean_point[0], mean_point[1]] = 1
                arrr[annot_pts[:, 0], annot_pts[:, 1]] = 0.5
            resized_annot = cv2.resize(arrr, (64, 64))
            dense_.append(resized_annot.copy())
            dense_mp_visualization.append(cv2.resize(arrr_mp_dense, (64, 64)).copy())
        sparse_ = []
        sparse_mean_points = []
        sparse_mp_visualization = []
        for entry in annotations:
            arrr = np.zeros(size_bf)
            arrr_mp = np.zeros(size_bf)
            s__ = []
            for ittem in annotations[entry]:
                points = annotations[entry][ittem]["range_angle"]["sparse"]
                annot_pts = np.array(points)
                mean_point = np.sum(np.array(points), axis=0) // len(np.array(points))
                s__.append(mean_point)
                # PRINT HERE? TO SEE THE POINTS? THEN CALCULATE THE MEAN POINT
                arrr[annot_pts[:, 0], annot_pts[:, 1]] = 0.5
                arrr_mp[mean_point[0], mean_point[1]] = 1
            sparse_mean_points.append(s__.copy())
            resized_annot = cv2.resize(arrr, (64, 64))
            sparse_.append(resized_annot.copy())
            sparse_mp_visualization.append(cv2.resize(arrr_mp, (64, 64)).copy())
        box_ = []
        for entry in annotations:
            arrr = np.zeros(size_bf)
            for ittem in annotations[entry]:
                points = annotations[entry][ittem]["range_angle"]["box"]
                annot_pts = np.array(points)
                arrr[annot_pts[:, 0], annot_pts[:, 1]] = 0.5
            resized_annot = cv2.resize(arrr, (64, 64))
            box_.append(resized_annot.copy())
        if differentiated:
            return (
                dense_[1:],
                sparse_[1:],
                box_[1:],
                sparse_mean_points[1:],
                sparse_mp_visualization[1:],
            )
        else:
            return dense_, sparse_, box_, sparse_mean_points, dense_mp_visualization

    def visualize_matrix(self, frame_name, selection=0):
        """
        Visualize matrix in notebook
        :param matrix:
        :param selection: 0 for angle, 1 for doppler
        :return:
        """

        ra_matrix, rd_matrix, annotations = self.load_data_from_frame(frame_name)
        fig, ax = plt.subplots()
        if selection == 0:
            cell_dists_map = ax.imshow(ra_matrix)
            fig.colorbar(ra_matrix)
        elif selection == 1:
            fig, ax = plt.subplots()
            cell_dists_map = ax.imshow(rd_matrix)
            fig.colorbar(rd_matrix)
        plt.show()


# seq_name = "2020-02-28-13-13-43"
# instances = ["000670", "000673"]
# frame_name = "000100"
# r = RadarLoader(seq_name)
# # r.display_color_image_stream(frame_name)
# r.display_range_angle_stream(frame_name)
# # r.display_range_doppler_stream(frame_name)
