# UNLABELED_Installation

Test PC
- Ubuntu 18.04
- CUDA GeForce RTX 2070


```
python run.py
```

If you want to test with video
```run.py
cap = cv2.VideoCapture("mov/test.mov")

img = frame
img = cv2.resize(img, (640, 480)) # Add
sized = cv2.resize(img, (yolo.width, yolo.height))
```
