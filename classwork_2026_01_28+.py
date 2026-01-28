import onnxruntime as ort
import torch
from torch.sparse import softmax
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
names_codes = ['хвора', 'здорова']

session = ort.InferenceSession("leukemia.onnx")

image = Image.open("data/lesson many/cells/UID_H13_15_3_hem.bmp")
# image.show()

input_tensor: torch.Tensor = test_transform(image)
input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.numpy()

results = session.run(
    None,
    input_feed={"input": input_tensor}
)

result = results[0][0]

ind = result.argmax()
name = class_names[ind]

result_tensor = torch.tensor(result)
softmax = torch.nn.Softmax()
probs = softmax(result_tensor).numpy()
prob = probs[ind]

print(f"На цьому малюнку з ймовірністю {prob*100}% зображена клітина {name} ({names_codes[ind]})." )







# print(input_tensor, input_tensor.shape)



