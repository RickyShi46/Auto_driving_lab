import cv2
import torch
import numpy as np

# Class for calculating safety distance


class SafetyDistance:

    def __init__(self, min_distance, middle_distance, line_threshold=3, flag2color=None):
        super(SafetyDistance, self).__init__()
        if flag2color is None:
            # bgr
            flag2color = {
                'danger': (0, 0, 255),
                'warning': (0, 255, 255),
                'safe': (0, 255, 0)
            }
        self.min_distance = min_distance
        self.middle_distance = middle_distance
        self.line_threshold = line_threshold
        self.flag2color = flag2color

    def compute_distance(self, img, pred_boxs):

        # Converting boxes to numpy arrays
        if pred_boxs is None or len(pred_boxs) == 0 or pred_boxs[0] is None:
            return None

        if isinstance(pred_boxs[0][1][0], torch.Tensor):
            pred_boxs = [(cls, torch.tensor(box)) for cls, box in pred_boxs]
            pred_boxs = [(cls, box.numpy()) for cls, box in pred_boxs]
        else:
            pred_boxs = [(cls, np.array(box)) for cls, box in pred_boxs]

        # Calculate the coordinates of the bottom midpoint of the image
        img_height = img.shape[0]
        img_width = img.shape[1]
        img_bottom_center_x = img_width // 2
        img_bottom_center_y = img_height

        # Calculate distance
        results = []
        for cls, box in pred_boxs:
            x1, y1, x2, y2 = box
            # Calculate the coordinates of the point in the box
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2
            # Calculate distance
            x = abs(img_bottom_center_x - box_center_x)
            y = abs(img_bottom_center_y - box_center_y)
            distance = np.sqrt(x ** 2 + y ** 2)
            results.append(
                (cls, box.tolist(), (box_center_x, box_center_y), distance))
        return results

    def is_safe_distance(self, distance_boxs):
        if distance_boxs is None or len(distance_boxs) == 0 or distance_boxs[0] is None:
            return None
        # Convert numpy arrays to raw
        pred_boxs = [(cls, tuple(box), tuple(center), int(dist))
                     for cls, box, center, dist in distance_boxs]
        # Determining if the distance is safe
        results = []
        for cls, box, center, dist in pred_boxs:
            if dist < self.min_distance:
                flag = 'danger'
            elif dist < self.middle_distance:
                flag = 'warning'
            else:
                flag = 'safe'
            results.append((cls, box, center, dist, flag))
        return results

    def draw_distance(self, img, distance_boxs):

        if distance_boxs is None or len(distance_boxs) == 0 or distance_boxs[0] is None:
            return None

        # Calculate the coordinates of the bottom midpoint of the image
        img_height = img.shape[0]
        img_width = img.shape[1]
        img_bottom_center_x = img_width // 2
        img_bottom_center_y = img_height
        # Draw the bottom centre of the image
        cv2.circle(img, (img_bottom_center_x, img_bottom_center_y),
                   25, (255, 0, 255), -1)
        # Draw two half circles
        cv2.circle(img, (img_bottom_center_x, img_bottom_center_y),
                   self.min_distance, self.flag2color['danger'], 2)
        cv2.circle(img, (img_bottom_center_x, img_bottom_center_y),
                   self.middle_distance, self.flag2color['warning'], 2)

        for cls, box, center, dist, flag in distance_boxs:
            color = self.flag2color[flag]
            center = (int(center[0]), int(center[1]))
            # Plotting target centroids
            cv2.circle(img, center, self.line_threshold * 3, color, -1)
            # Line between the centre of the target and the centre of the bottom
            cv2.line(img, center, (img_bottom_center_x,
                     img_bottom_center_y), color, self.line_threshold)

    def process(self, img, pred_boxs):
        results = self.compute_distance(img, pred_boxs)
        results = self.is_safe_distance(results)
        return results
