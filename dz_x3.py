# # Завдання 1
# # Відкрийте відео з файлу data\lesson8\meetings.mp4
# # Застосуйте детекцію та виведіть результат, підберіть параметри
# # Можете змінити розмір кадру для кращої візуалізації
# # cv2.resize()
#
# # детекція об'єктів(YOLO)
# import ultralytics
# import cv2
#
# model = ultralytics.YOLO('yolov8s.pt')
#
# cap = cv2.VideoCapture('data/lesson8/meetings.mp4')
#
# if not cap.isOpened():
#     print("❌ Відео не відкривається!")
#     exit()
#
# while True:
#     success, img = cap.read()
#     if not success:
#         break
#
#     img = cv2.resize(img, None, fx=0.4, fy=0.4)
#
#     results = model.predict(
#         img,
#         conf=0.3,
#         iou=0.7
#     )
#
#     result = results[0]
#     res_img = result.plot()
#     cv2.imshow('result', res_img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # :)
#         break
#
# cv2.waitKey(0)

# Завдання 2
# Відкрийте відео з файлу data\lesson8\meetings.mp4
# Застосуйте детекцію та почніть показувати відео з моменту, коли людей стало 5

import ultralytics
import cv2

model = ultralytics.YOLO('yolov8s.pt')
cap = cv2.VideoCapture('data/lesson8/meetings.mp4')

threshold = 5
started = False

while True:
    ok, img = cap.read()
    if not ok:
        break

    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    r = model(img, conf=0.3, iou=0.7)[0]

    people = sum(int(c) == 0 for c in r.boxes.cls)
    if people >= threshold:
        started = True

    if started:
        cv2.imshow('result', r.plot())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
