import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import numpy as np


test_transform = transforms.Compose(
    [transforms.Resize([200, 200]),
    transforms.CenterCrop(180),
    transforms.ToTensor()
    ]
)

class_names = ['all', 'hem']

session = ort.InferenceSession("leukemia.onnx")

image = Image.open("data/lesson many/cells/UID_H13_11_5_hem.bmp")
# image.show()

input_tensor = test_transform(image)
print(input_tensor, input_tensor.shape)



