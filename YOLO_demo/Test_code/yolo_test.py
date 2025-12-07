#run YOLO-You-Only-Live-Once/YOLO_demo


import os
from ultralytics import YOLO
print(f"current dir: {os.getcwd()}")
# model_path = "./runs/detect/train2/weights/best.pt"
model_path_fine_tuned = "../runs/detect/train5/weights/best.pt"
model = YOLO(model_path_fine_tuned)  

"""
Prediciton 3,4,5 is the yolov8n.pt model and prediction 1,2 is the best.pt model
"""

# # option 1 see one image <-try this first compare two models!



# # image_path = "images/val/game_20251122_BPrP-O6-_00750.jpg"
# image_path = "images/tt/img9.jpg"
# results = model(image_path)   


# for r in results:
#     r.show()        #show in screen not save




#option2 see all path



f#