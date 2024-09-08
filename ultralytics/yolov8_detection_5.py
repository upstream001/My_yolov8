# from ultralytics import YOLO
#
# model=YOLO('yolov8n.pt')
#
# model.train(data='data.yaml',workers=0,epochs=50,batch=4)

# model=YOLO('runs/detect/train4/weights/best.pt')
#
# result=model.predict('datasets/HatSample/test/images/000034_jpg.rf.8q07ltxA7LKxAn4jiJeu.jpg',save=True)
import time
import cv2
from ultralytics import YOLO
# -------------------------------------------------------------

# model=YOLO('yolov8s-seg.pt')
#
# video_path='6387-191695740_large.mp4'
# cap=cv2.VideoCapture(video_path)
#
# while cap.isOpened():
#     success,frame=cap.read()
#     if success:
#         time_start=time.time()
#         result=model(frame)
#         time_end=time.time()
#         total_time=time_end-time_start
#         fps=1/total_time
#         annotated_frame=result[0].plot()
#         cv2.putText(annotated_frame,f'fps:{int(fps)}',(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), 2)
#         cv2.imshow('Inference pic',annotated_frame)
#         if cv2.waitKey(1)&0xFF==ord('q'):
#             break
#     else:
#         break
# cap.release()
# cv2.destroyAllWindows()

# -------------------------------------------------------------


import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO


class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device:', self.device)

        self.model = self.load_model()

    def load_model(self):
        model = YOLO('yolov8n.pt')
        model.fuse()  # 可以将卷积层（Conv）和批量归一化层（BatchNorm）融合成单个层，这样可以减少计算量并加速推理过程。
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []
        for result in results:
            boxes=result.boxes.cpu().numpy()
            xyxys=boxes.xyxy
            for xyxy in xyxys:
                cv2.rectangle(frame,(int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])),(0,255,0))
            # xyxys.append(boxes.xyxy)
            # confidences.append(boxes.conf)
            # class_ids.append(boxes.cls)

        return frame

    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:

            start_time = time()

            ret, frame = cap.read()
            assert ret

            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Detection', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(capture_index='6387-191695740_large.mp4')
detector()