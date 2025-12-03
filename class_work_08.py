from tabnanny import verbose

import cv2
import ultralytics
import numpy as np

model = ultralytics.YOLO('yolo11s-pose.pt')
#
# img = cv2.imread('data/lesson_pose/human.jpg')
#
# results = model.predict(img)
# result = results[0]
#
# res_img = result.plot()
#
# # ключові точки
# keypoints = result.keypoints
#
# # ймовірності для кожної точки
# conf = keypoints.conf
#
# # print(conf)
# # print(conf.shape)  # (кількість людей, кількість точок(17))
#
# # координати xy
# xy = keypoints.xy
#
# # print(xy)
# # print(xy.shape) # (кількість людей, кількість точок(17), координати)
#
# # координати правої долоні
# xy_right_hand = xy[0, 10]  # людина 0, точка 10
# xy_right_hand = xy_right_hand.cpu()  # відключити від графічного процесора
# xy_right_hand = xy_right_hand.numpy()  # перевести у звичайний масив
#
# x, y = xy_right_hand
#
# # треба перевести в int
# x = int(x)
# y = int(y)
#
# # print(x)
# # print(y)
#
# # намалювати коло на зображення
# cv2.circle(
#     img,   # зображення де малювати коло
#     (x, y),     # координати центру
#     20,         # радіус кола
#     (255, 0, 0),  # колір у bgr(тут синій)
#     -1                 # товщина лінії(-1 означає повністю заповнене коло)
# )
#
# # накласти текст на зображення
# cv2.putText(
#     img,                  # зображення
#     'Right hand',    # текст
#     (x+30, y-30),     # нижня ліва точка початку тексту
#     cv2.FONT_HERSHEY_SIMPLEX,    # шрифт
#     0.8,          # розмір шрифту(відсоток до стандарту)
#     (0, 0, 0),      # колір у bgr(тут чорний)
#     2           # товщина ліній
#
# )
#
# xy = xy.cpu().numpy()
# # ліва стопа
# x_left_foot, y_left_foot = xy[0, 15]
#
# # праве плече
# x_right_shoulder, y_right_shoulder = xy[0, 6]
#
# # чи справді праве плече знаходиться правіше за ліву стопу
# if x_right_shoulder > x_left_foot:
#     print("праве плече знаходиться правіше за ліву стопу")
# else:
#     print("праве плече знаходиться лівіше за ліву стопу")
#
# # чи справді праве плече знаходиться вище за ліву стопу
# if y_right_shoulder < y_left_foot:
#     print("праве плече знаходиться вище за ліву стопу")
# else:
#     print("праве плече знаходиться нижче за ліву стопу")
#
# cv2.imshow('original', img)
# cv2.imshow('result', res_img)
# cv2.waitKey(0)

# Модуль 2. Комп’ютерний зір
# Тема: opencv. Частина 3
# Завдання 1
# =============================================================================
# Відкрийте відео data/lesson_pose/sitting.mp4
# Ваша задача рахувати кількість присідань.
# Отримайте перший кадр та виділіть основні точки.
# Отримайте координати однієї з долонь та лівого коліна.
# Вважайте що людина присіла, коли її рука опустилась нижче коліна, і піднялась коли її рука опинилась вище коліна.
# Оскільки на відео є декілька людей то обирайте ту, яка
# знаходиться найближче, тобто в якої найбільша площа
# рамки(можете потренуватись на 200-му кадрі)

orig_video = cv2.VideoCapture('data/lesson_pose/sitting.mp4')
move_down = True
counter = 0

# #шукаэмо 200 кадр
# for i in range(200):
#     success, img = orig_video.read()

# model_results = model.predict(img, verbose=False)
# model_result = model_results[0]

def get_max_index(model_result):

    xy_boxes = model_result.boxes.xywh
    # print(xy_boxes)
    # print(xy_boxes.shape)

    # дістаємо ширину і висоту кожної рамки (людини)
    width = xy_boxes[:,2]
    height = xy_boxes[:,3]

    # print(width)
    # print(height)

    area = width * height
    # print(area)

    area = area.numpy()
    max_index = np.argmax(area)

    return max_index


# plot_img = model_result.plot()
# cv2.imshow('plot_img', plot_img)
#
#
# cv2.waitKey(0)

while True:
    success, img = orig_video.read()

    if not success:
        break

    model_results = model.predict(img, verbose=False)
    model_result = model_results[0]
    max_index = get_max_index(model_result)

    # plot_img = model_result.plot()
    # cv2.imshow('plot_img', plot_img)

    keypoints = model_result.keypoints[max_index]
    xy = keypoints.xy
    # print(xy.shape)

    left_hand = xy[0, 9]
    y_left_hand = left_hand[1]

    right_hand = xy[0, 10]
    y_right_hand = right_hand[1]

    left_knee = xy[0, 13]
    y_left_knee = left_knee[1]

    right_knee = xy[0, 14]
    y_right_knee = right_knee[1]

    # (беремо коліно, ЯКЕ ВИЩЕ)
    y_knee_min = min(y_left_knee, y_right_knee)

    # человек присел
    if y_left_hand > y_knee_min and move_down:
        move_down = False
        counter += 0.5

    if y_left_hand < y_knee_min and not move_down:
        move_down = True
        counter += 0.5

    cv2.putText(
        img,                  # зображення
        f'Counter: {counter}, Max_index: {max_index}',    # текст
        (20, 20),     # верхний левый угол
        cv2.FONT_HERSHEY_SIMPLEX,    # шрифт
        0.8,          # розмір шрифту(відсоток до стандарту)
        (255, 255, 255),      # колір у bgr(тут чорний)
        2           # товщина ліній

    )

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Завдання 2
# Відкрийте відео data/lesson_pose/hopak.mp4
# Покажіть відео добавляючи на кожен кадр назву руху
# Практичне завдання
# Figure 2 Руки в сторону
# Figure 1 Руки в
# боки
# Figure 3 Одна рука
# вгору
# Завдання 3
# Відкрийте відео data/lesson_pose/hands.mp4
# Покажіть відео змінюючи яскравість відео, в залежності
# від відстані між долонями. За одиницю виміру візьміть
# відстань між плечима

