# сегментація зображень
import ultralytics
import numpy as np
import cv2

# модель для сегментації
model = ultralytics.YOLO('yolo11s-seg.pt')

img = cv2.imread('C:\AI01\ITStep-AI\data\lesson_seg\human.jpg')

# # застосування моделі
# # results -- list з результами для кожного зображення в predict
# results = model.predict(
#     img,
#     # conf=0.5,
#     # iou=0.1
# )
#
# # дістати результати для першого(єдиного) зображення
# result = results[0]
#
# # візуалізація результату
# res_img = result.plot(
#     boxes=True,   # чи показувати рамки(boxes)
#     masks=True,    # чи показувати маски сегментації(boxes)
# )
#
# # маски об'єктів
# masks = result.masks.data
# # print(masks)
# # print(masks.shape)  # (кількість об'єктів, висота, ширина)
#
# # класи об'єктів
# cls = result.boxes.cls
# print(cls)
#
# # назви класів
# names = result.names
# print(names)
#
# # де знаходиться людина
# idx = 0
# # маска людини
# mask = masks[idx]
# # print(mask)
# # print(mask.shape)
#
# # переведення маски у формат opencv
# mask = mask.cpu()   # відключення від графічного процесора
# mask = mask.numpy() # переведення у масив numpy
# mask = mask.astype(np.uint8)   # зміна типу даних
# mask *= 255         # заміна 1 на 255 щоб було видно
#
#
# # print(mask)
# # print(mask.dtype)
# # print(np.unique((mask)))  # унікальні значення в масиві


# відео
cap = cv2.VideoCapture(0)

# зображення фону
background_img = cv2.imread('C:\AI01\ITStep-AI\data\lesson4\canal.png')
cv2.imshow('background', background_img)

# зміна розміру зображення з фоном
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

background_img = cv2.resize(background_img, (width, height))

while True:
    success, img = cap.read()

    if not success:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # застосувати модель
    results = model.predict(img)
    result = results[0]

    res_img = result.plot()

    # маска
    masks = result.masks.data
    mask = masks[0]
    mask = mask.cpu()
    mask = mask.numpy()
    mask = mask.astype(np.uint8)
    mask *= 255

    # добавити фон
    mask_bool = mask.astype(bool)

    # заміна пікселів які не людина
    img[~mask_bool] = background_img[~mask_bool]

    cv2.imshow('res', res_img)
    cv2.imshow('mask', mask)
    cv2.imshow('original', img)

    # print(f'{img.shape = }')
    # print(f'{mask.shape = }')
    # print(f'{background_img.shape = }')

# cv2.imshow('mask', mask)
# cv2.imshow('result', res_img)
# cv2.imshow('original', img)
# cv2.waitKey(0)