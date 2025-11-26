# Завдання 1
# Отримайте перший кадр з файлу data\lesson8\animals.mp4 та виведіть його на екран.

# Змініть параметри моделі conf та iou і подивіться як це впливає на результат.
# Отримайте рамки для кожного об’єкта, виріжіть їх та виведіть як окремі зображення

# # детекція об'єктів(YOLO)
#
import ultralytics
import cv2
from torch.xpu import device

#
# # створення моделі
model = ultralytics.YOLO('yolov8s.pt')
#
# # отримати зображення з ввідео
cap = cv2.VideoCapture('animals.mp4')
success, img = cap.read()
#
img = cv2.resize(img, None, fx=0.5, fy=0.5)
#
cv2.imshow('original', img)
#
# # застосування моделі
# Проведіть детекцію об’єктів зо допомогою YOLO та виведіть результати.
# # predict може отримувати декілька зображень за раз
# # predict([img1, img2, img3]) і тому results це список
# # [result1, result2, result3]
#
results = model.predict(
    img,
    conf=0.25,       # мінімальній відсоток для об'єкта щоб попасти в result
    iou=0.8      # максимально можливий рівень перетину рамок(якщо більше то вважаємо
)
 #
#

# # results -- список з одним елементом
result = results[0]
image_result = result.plot()
cv2.imshow('image result', image_result)



#
# # отримати назви класів(об'єкти які вмієме визначати модель)
# names = result.names # dict --- id: str
# print(names)
#
# # самі об'єкти
# cls = result.boxes.cls
# print(cls)
#
# # візуалізація результів
# res_img = result.plot()
# cv2.imshow('result', res_img)
#
#
# # ймовірності
# conf = result.boxes.conf
# print(conf)
#
# Отримайте рамки для кожного об’єкта, виріжіть їх та виведіть як окремі зображення
# # рамка(box)
boxes = result.boxes
#
# # отримати перший об'єкт

#
# print(boxes)
for i in range(len(boxes)):
    xyxy = boxes[i].xyxy
    print(xyxy[0])
    # print(xyxy[0].shape)
    x1, y1, x2, y2 = map(int, xyxy[0])
    print(x1, y1, x2, y2)

    roi = img[y1:y2, x1:x2]
    cv2.imshow(f'roi-{i}', roi)







#
# # координати
# xyxy = box.xyxy[0]
#
# # переведення координат в int
# x1, y1, x2, y2 = map(int, xyxy)
#
# # вирізати об'єк з всього зображення
# # y -- відповідають за рядки
# # x -- відповідають за стовпчики
# roi = img[y1:y2, x1:x2]
#
# cv2.imshow('object', roi)
#
#
# # # відео
# # while True:
# #     success, img = cap.read()
# #
# #     if not success:
# #         break
# #
# #     img = cv2.resize(img, None, fx=0.5, fy=0.5)
# #
# #     results = model.predict(img)
# #     result = results[0]
# #
# #     res_img = result.plot()
# #
# #     cv2.imshow('result', res_img)
# #
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
#
#
cv2.waitKey(0)