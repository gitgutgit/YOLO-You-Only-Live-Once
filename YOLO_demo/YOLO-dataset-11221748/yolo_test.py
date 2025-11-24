import os
from ultralytics import YOLO
print(f"current dir: {os.getcwd()}")
# model_path = "./runs/detect/train2/weights/best.pt"
model_path_yolov8n = "yolov8n.pt"
model_path_fine_tuned = "best.pt"
model = YOLO(model_path_fine_tuned)  

"""
Prediciton 3,4,5 is the yolov8n.pt model and prediction 1,2 is the best.pt model
"""

# option 1 see one image <-try this first compare two models!



image_path = "images/val/game_20251122_BPrP-O6-_00750.jpg"
results = model(image_path)   


for r in results:
    r.show()        #show in screen not save




#option2 see all path



# folder_path = "./images/val"
# results = model(folder_path, 
#                 project="demo_test_result",  # save in this folder
#                 name="predictions",          # sub folder name (if u want to save in different folder)
#                 save=True) 

# # limit to 30 images (for testing but could be changed if u want to see all images!)
# if len(results)>30:
#     results = results[:30]
# else:
#     pass

# # show and save
# for r in results:
#     r.show()        #show in screen
