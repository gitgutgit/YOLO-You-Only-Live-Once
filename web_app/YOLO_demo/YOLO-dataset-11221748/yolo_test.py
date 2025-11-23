import os
from ultralytics import YOLO
print(f"current dir: {os.getcwd()}")
# model_path = "./runs/detect/train2/weights/best.pt"
model_path = "best.pt"
model = YOLO(model_path)  

# option 1 see one image
# image_path = "images/val/game_20251122_BPrP-O6-_00790.jpg"
# results = model(image_path)   


# for r in results:
#     r.show()        #show in screen




# option2 see all path
folder_path = "./images/val"
results = model(folder_path, 
                project="demo_test_result",  # save in this folder
                name="predictions",          # sub folder name (if u want to save in different folder)
                save=True) 

# limit to 30 images (for testing but could be changed if u want to see all images!)
if len(results)>30:
    results = results[:30]
else:
    pass

# show and save
for r in results:
    r.show()        #show in screen
