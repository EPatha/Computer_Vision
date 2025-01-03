import cv2
import torch
from torchvision import transforms

# Load the pre-trained ESRGAN model (download or use the pre-trained weights)
model = torch.hub.load('xinntao/ESRGAN', 'ESRGAN')

# Load your image
img = cv2.imread("low_res_image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = transforms.ToTensor()(img).unsqueeze(0)  # Convert to tensor

# Perform super-resolution
with torch.no_grad():
    output = model(img)

# Convert tensor back to image
output_image = output.squeeze().numpy().transpose(1, 2, 0)
output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

# Save or display the enhanced image
cv2.imwrite("high_res_image.jpg", output_image)
