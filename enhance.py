import cv2
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda")
print(f"Using: {torch.cuda.get_device_name(0)}")

def enhance_frame(frame):
    img = frame.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    upscaled = F.interpolate(tensor, scale_factor=2, mode='bicubic', align_corners=False)

    blurred = F.avg_pool2d(upscaled, kernel_size=3, stride=1, padding=1)
    sharpened = upscaled + 0.5 * (upscaled - blurred)
    sharpened = torch.clamp(sharpened, 0, 1)

    result = sharpened.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return (result * 255).astype(np.uint8)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    enhanced = enhance_frame(frame)

    cv2.imshow("Original", frame)
    cv2.imshow("Enhanced", enhanced)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()