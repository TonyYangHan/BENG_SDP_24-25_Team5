from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.yaml", verbose = False)
model.train(data="YOLO/data.yaml", epochs=5, verbose = False, batch = 3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
result = model("../train/images/test22.jpg", verbose = False)
result[0].save("YOLO/test22_anno.jpg") # Get the rendered image (with bounding boxes)