# Тема: opencv. Частина 3
#  Завдання 1
# Відкрийте зображення data/lesson_seg/crop3.jpg  Проведіть сегментацію зображення використовуючи
# модель data/lesson_seg/crop-seg.jpg
# Покажіть усі маски рослин з підписами назви цієї
# рослини.  Покажіть також самі рослини, для цього застосуйте
# маску, і всі зайві пікселі замініть на 255(зробити білий фон)

import ultralytics
import numpy as np
import cv2

model = ultralytics.YOLO('data/lesson_seg/crop-seg.pt')

img = cv2.imread('data/lesson_seg/crop3.jpg')

cv2.imshow('original', img)

results = model.predict(
    img
    )

result = results[0]

res_img = result.plot(
    boxes=True,
    masks=True
)

cv2.imshow('segments', res_img)

# Покажіть усі маски рослин з підписами назви цієї рослини.
names = result.names
print(names)

masks = result.masks.data
print(masks[0])

# print(mask)

cls = result.boxes.cls
print(cls)

nums = len(cls)

heigh, width, _  = img.shape
# print(heigh, width)

for i in range(nums):
    name = names[int(cls[i])]

    mask = masks[i]
    mask = mask.numpy()
    mask = mask.astype(np.uint8)
    mask *= 255
    mask = cv2.resize(mask, (width, heigh))

    # cv2.imshow(f'{name}, {i}', mask)
    # print(mask.shape)

    mask_bool = mask.astype(bool)
    img_current = img.copy()
    img_current[~mask_bool] = 255

    cv2.imshow(f'{name}, {i}', img_current)


#
# print(img.shape)

cv2.waitKey(0)

# Покажіть також самі рослини, для цього застосуйте маску, і всі зайві пікселі замініть на 255(зробити білий фон)


