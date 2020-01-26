import cv2
import os
import dlib
import numpy as np
import matplotlib.pyplot as plt

def smile_detection(device_num, dir_path, basename, ext='.jpg', delay=1, window_name='frame'):
    predictor_path = "predictor/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    cap = cv2.VideoCapture(device_num)

    if not cap.isOpened():
        return

    n = 0
    resize_rate = 4
    privacy_mask = 1
    show_smile = 1
    values_timeseries = np.zeros(10)
    value = 0
    print(values_timeseries)

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    while True:
        ret, frame = cap.read()
        height, width = frame.shape[:2]
        temp_frame = cv2.resize(frame, (int(width/resize_rate), int(height/resize_rate)))
        dets = detector(temp_frame, 1) # face detection

        for k, d in enumerate(dets):
            shape = predictor(temp_frame, d)
            # drawing the parts of face
            rect_offset = 20
            if privacy_mask == 1:
                cv2.rectangle(frame, (int(d.left() * resize_rate) - rect_offset, int(d.top() * resize_rate) - rect_offset), \
                    (int(d.right() * resize_rate) + rect_offset, int(d.bottom() * resize_rate) + rect_offset), (255, 255, 255), -1)

            for shape_point_count in range(shape.num_parts):
                shape_point = shape.part(shape_point_count)
                if show_smile == 1:
                    if ( (shape_point_count == 48) or (shape_point_count == 54) or (shape_point_count == 60) or
                       (shape_point_count == 64) or (shape_point_count == 8) or (shape_point_count == 27) or (shape_point_count == 66) ):
                        cv2.circle(frame, (int(shape_point.x * resize_rate), int(shape_point.y * resize_rate)), 2, (0, 0, 255), -1)
                        value = np.sqrt(np.power(shape.part(48).x * resize_rate - shape.part(66).x * resize_rate, 2) + np.power(shape.part(48).y * resize_rate - shape.part(66).y * resize_rate, 2))
                        values_timeseries = np.append(np.delete(values_timeseries, 0), value)
                else:
                    if shape_point_count < 17: # [0-16]:輪郭
                        cv2.circle(frame, (int(shape_point.x * resize_rate), int(shape_point.y * resize_rate)), 2, (0, 0, 255), -1)
                    elif shape_point_count < 22: # [17-21]眉（右）
                        cv2.circle(frame, (int(shape_point.x * resize_rate), int(shape_point.y * resize_rate)), 2, (0, 255, 0), -1)
                    elif shape_point_count < 27: # [22-26]眉（左）
                        cv2.circle(frame, (int(shape_point.x * resize_rate), int(shape_point.y * resize_rate)), 2, (255, 0, 0), -1)
                    elif shape_point_count < 31: # [27-30]鼻背
                        cv2.circle(frame, (int(shape_point.x * resize_rate), int(shape_point.y * resize_rate)), 2, (0, 255, 255), -1)
                    elif shape_point_count < 36: # [31-35]鼻翼、鼻尖
                        cv2.circle(frame, (int(shape_point.x * resize_rate), int(shape_point.y * resize_rate)), 2, (255, 255, 0), -1)
                    elif shape_point_count < 42: # [36-4142目47）
                        cv2.circle(frame, (int(shape_point.x * resize_rate), int(shape_point.y * resize_rate)), 2, (255, 0, 255), -1)
                    elif shape_point_count < 48: # [42-47]目（左）
                        cv2.circle(frame, (int(shape_point.x * resize_rate), int(shape_point.y * resize_rate)), 2, (0, 0, 128), -1)
                    elif shape_point_count < 55: # [48-54]上唇（上側輪郭）
                        cv2.circle(frame, (int(shape_point.x * resize_rate), int(shape_point.y * resize_rate)), 2, (0, 128, 0), -1)
                    elif shape_point_count < 60: # [54-59]下唇（下側輪郭）
                        cv2.circle(frame, (int(shape_point.x * resize_rate), int(shape_point.y * resize_rate)), 2, (128, 0, 0), -1)
                    elif shape_point_count < 65: # [60-64]上唇（下側輪郭）
                        cv2.circle(frame, (int(shape_point.x * resize_rate), int(shape_point.y * resize_rate)), 2, (0, 128, 255), -1)
                    elif shape_point_count < 68: # [65-67]下唇（上側輪郭）
                        cv2.circle(frame, (int(shape_point.x * resize_rate), int(shape_point.y * resize_rate)), 2, (128, 255, 0), -1)

            cv2.putText(frame, "Current : " + str(value), (0, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(frame, "Average : " + str(np.mean(values_timeseries)), (0, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

        # ax2.cla()
        # ax2.hist(values_timeseries, alpha=0.5)
        # ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), aspect='auto')
        # plt.pause(.01)

        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(dir_path + "/" + basename + str(n) + ext, frame)
            print(dir_path + "/" + basename + str(n) + ext)
            n = n + 1
            print(n)

    cv2.destroyWindow(window_name)

if __name__ == '__main__':
    smile_detection(0, 'data/temp', 'camera_capture')
