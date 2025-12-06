# Тяжко працювати з зображенням, розмір якого 1066x1920. Тому створив відео squat_smal (удвічі менше,
# і вже далі працював з ним). Теоретично, можна було б відображати оригінальне відео в меншому вікні,
# але я не став цього робити, щоб не ускладнювати код. + задовбусь ключові точки перераховувати // 2
# import cv2

# SCALE = 2  # коефіцієнт масштабування
#
# input_file = 'data/lesson_pose/squat.mp4'
# output_file = 'data/lesson_pose/scale_squat.mp4'
#
# # Відкрити відео
# cap = cv2.VideoCapture(input_file)
#
# # Отримати параметри відео
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
#
# # Підготовка для запису масштабованого відео
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# writer = cv2.VideoWriter(
#     output_file,
#     fourcc,
#     fps,
#     (width // SCALE, height // SCALE),
#     isColor=True  # кадри кольорові
# )

# Обробка та збереження відео
# while True:
#     success, frame = cap.read()
#     if not success:
#         break
#
#     # Масштабування кадру
#     frame_small = cv2.resize(frame, (width // SCALE, height // SCALE))
#     writer.write(frame_small)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# writer.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from ultralytics import YOLO
import utils   # get_angle()

# --- Файли ---
video_path = 'data/lesson_pose/squat_smal.mp4'
model = YOLO('yolo11s-pose.pt')

# --- Межі ---
UPPER_ANGLE = 160
LOWER_ANGLE = 90

# --- Статус ---
squat_count = 0
down = False

# --- Відео ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Не вдалося відкрити відео")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO
    results = model(frame, verbose=False)[0]

    if results.keypoints is None or len(results.keypoints) == 0:
        cv2.imshow("Squat Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Беремо першу людину і перетворюємо в (17,3)
    kp = results.keypoints[0].data.numpy()[0]
    # print(kp.shape)

    # Перевірка на наявність всіх ключових точок
    if kp.shape[0] != 17:
        continue

    x1, y1 = kp[12][:2]  # ПРАВЕ стегно
    x2, y2 = kp[14][:2]  # ПРАВЕ коліно
    x3, y3 = kp[16][:2]  # ПРАВА щиколотка

    # кут
    angle = utils.get_angle(x1, y1, x2, y2, x3, y3)

    # кількість присідань
    if angle < LOWER_ANGLE and not down:
        down = True
        squat_count += 0.5  # присіла, додаємо пів

    if angle > UPPER_ANGLE and down:
        down = False
        squat_count += 0.5  # встав, додаємо ще пів

    # --- Малювання ---
    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.line(frame, (int(x2), int(y2)), (int(x3), int(y3)), (255, 0, 0), 2)

    cv2.circle(frame, (int(x1), int(y1)), 7, (0,255,255), -1)
    cv2.circle(frame, (int(x2), int(y2)), 7, (0,255,255), -1)
    cv2.circle(frame, (int(x3), int(y3)), 7, (0,255,255), -1)

    cv2.putText(frame, f"Angle: {int(angle)}", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    cv2.putText(frame, f"Squats: {squat_count}", (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    cv2.imshow("Squat Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
