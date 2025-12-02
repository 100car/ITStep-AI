# Завдання 1
# Відкрийте відео з файлу data\lesson7\meter.mp4.
# Проведіть бінарізацію кадрів та збережіть в новий файл.
# Можливо очистіть від шуму або наведіть різкість через bilateralFilter

import cv2
import numpy as np

cap = cv2.VideoCapture('lesson7/meter.mp4')

if not cap.isOpened():
    print("❌ Відео не відкривається!")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('result.mp4', fourcc, fps, (w, h), isColor=True)

while True:
    success, img = cap.read()
    if not success:
        break

    # 1. Розмиття
    blur = cv2.bilateralFilter(img, d=7, sigmaColor=75, sigmaSpace=75)
    # blur = cv2.bilateralFilter(img, d=9, sigmaColor=100, sigmaSpace=100)

    # 2. Грейскейл
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # 3. Адаптивна бінарізація
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 5
    )

    # 4. Морфологія (більше ядро + open+close)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    # Для запису
    clean_bgr = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
    out.write(clean_bgr)

    # ----------- ПОРІВНЯННЯ 2×2 -----------

    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    clean_bgr = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)

    half_w, half_h = w // 2, h // 2
    im_ori  = cv2.resize(img,        (half_w, half_h))
    im_blur = cv2.resize(blur,       (half_w, half_h))
    im_bin  = cv2.resize(binary_bgr, (half_w, half_h))
    im_cln  = cv2.resize(clean_bgr,  (half_w, half_h))

    top = cv2.hconcat([im_ori, im_blur])
    bottom = cv2.hconcat([im_bin, im_cln])
    combined = cv2.vconcat([top, bottom])

    cv2.imshow("Original | Blur | Binary | Clean (2x2)", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
