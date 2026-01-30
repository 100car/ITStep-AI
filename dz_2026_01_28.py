import random
import onnxruntime as ort
import torch
from torchvision import transforms
from PIL import Image

IMG_SIZE = 64
IMAGE_DIR = "data/lesson many/fruits/"

test_transform = transforms.Compose(
    [
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
    ]
)

class_names = ['Apple Braeburn',
 'Apple Granny Smith',
 'Apricot',
 'Avocado',
 'Banana',
 'Blueberry',
 'Cactus fruit',
 'Cantaloupe',
 'Cherry',
 'Clementine',
 'Corn',
 'Cucumber Ripe',
 'Grape Blue',
 'Kiwi',
 'Lemon',
 'Limes',
 'Mango',
 'Onion White',
 'Orange',
 'Papaya',
 'Passion Fruit',
 'Peach',
 'Pear',
 'Pepper Green',
 'Pepper Red',
 'Pineapple',
 'Plum',
 'Pomegranate',
 'Potato Red',
 'Raspberry',
 'Strawberry',
 'Tomato',
 'Watermelon']

class_number = len(class_names)

picture_codes = [f"{i:02d}" for i in range(class_number)]
names_codes = [f"{name} ({code})" for name, code in zip(class_names, picture_codes)]

for _ in range(5):
    num = str(random.randint(0, 66))
    image_name = IMAGE_DIR + num + ".jpg"
    # print(image_name)

    session = ort.InferenceSession("fruits+.onnx")

    image = Image.open(image_name)
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
    softmax = torch.nn.Softmax(dim=0)
    probs = softmax(result_tensor).numpy()
    prob = probs[ind]

    print(f"На малюнку №{num} з ймовірністю {prob*100:.2f}% зображено {names_codes[ind]}." )
