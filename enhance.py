import cv2
import torch
import numpy as np
import sys
sys.path.insert(0, r'C:\Users\ASUS\Real-ESRGN')
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

device = torch.device("cuda")
print(f"Using: {torch.cuda}")

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

upsampler = RealESRGANer(
    scale=4,
    model_path="weights/RealESRGAN_x4plus.pth",
    model=model,
    tile=128,
    tile_pad=10,
    pre_pad=0,
    half=True
)
print("Model loaded!")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output, _=upsampler.enhance(frame, outscale=2)

    cv2.imshow('original', frame)
    cv2.imshow('Enhanced', output)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()