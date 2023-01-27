import argparse
import cv2
import numpy as np
import onnxruntime as ort

lane_colors = [(68,65,249),(44,114,243),(30,150,248),(74,132,249),(79,199,249),(109,190,144),(142, 144, 77),(161, 125, 39)]
log_space = np.logspace(0,2, 50, base=1/10, endpoint=True)

class LSTR():
    def __init__(self):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession("lstr_360x640.onnx", so)
        model_inputs = self.session.get_inputs()
        self.rgb_input_name = model_inputs[0].name
        self.mask_input_name = model_inputs[1].name
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.mask_tensor = np.zeros((1, 1, self.input_height, self.input_width), dtype=np.float32)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1).T

    def draw_lanes(self, input_img, detected_lanes, good_lanes):
        # Write the detected line points in the image
        visualization_img = input_img.copy()

        # Draw a mask for the current lane
        right_lane = np.where(good_lanes == 0)[0]
        left_lane = np.where(good_lanes == 5)[0]

        if (len(left_lane) and len(right_lane)):
            lane_segment_img = visualization_img.copy()

            points = np.vstack((detected_lanes[left_lane[0]].T, np.flipud(detected_lanes[right_lane[0]].T)))
            cv2.fillConvexPoly(lane_segment_img, points, color=(0, 191, 255))
            visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)

        for lane_num, lane_points in zip(good_lanes, detected_lanes):
            for lane_point in lane_points.T:
                cv2.circle(visualization_img, (lane_point[0], lane_point[1]), 3, lane_colors[lane_num], -1)

        return visualization_img

    def detect(self, frame):
        img_height, img_width = frame.shape[:2]
        img = cv2.resize(frame, (self.input_width, self.input_height))

        img = (img.astype(np.float32) / 255.0 - self.mean) / self.std

        img = img.transpose(2, 0, 1)
        input_tensor = img[np.newaxis, :, :, :].astype(np.float32)

        # Inference
        outputs = self.session.run(self.output_names, {self.rgb_input_name: input_tensor, self.mask_input_name: self.mask_tensor})

        ## process outputs
        pred_logits = outputs[0]
        pred_curves = outputs[1]

        # Filter good lanes based on the probability
        prob = self.softmax(pred_logits)
        good_detections = np.where(np.argmax(prob, axis=-1) == 1)
        pred_logits = pred_logits[good_detections]
        pred_curves = pred_curves[good_detections]

        lanes = []
        for lane_data in pred_curves:
            bounds = lane_data[:2]
            k_2, f_2, m_2, n_1, b_2, b_3 = lane_data[2:]

            # Calculate the points for the lane
            # Note: the logspace is used for a visual effect, np.linspace would also work as in the original repository
            y_norm = bounds[0] + log_space * (bounds[1] - bounds[0])
            x_norm = (k_2 / (y_norm - f_2) ** 2 + m_2 / (y_norm - f_2) + n_1 + b_2 * y_norm - b_3)
            lane_points = np.vstack((x_norm * img_width, y_norm * img_height)).astype(int)

            lanes.append(lane_points)

        return lanes, good_detections[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='images/2.jpg', help="image path")
    args = parser.parse_args()

    net = LSTR()
    srcimg = cv2.imread(args.imgpath)
    detected_lanes, lane_ids = net.detect(srcimg)
    dstimg = net.draw_lanes(srcimg, detected_lanes, lane_ids)

    winName = 'Deep learning lane detection in ONNXRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
