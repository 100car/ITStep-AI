# Завдання 1
# Відкрийте зображення data\lesson2\darken.png. Проведіть з ним наступні операції, переведіть його в HSV формат та
# обробіть канал Value наступними способами:
#  застосуйте вирівнювання гістограм
#  збільшіть значення десь на 20-50%, оскільки тут результат буде типу float32 та явно вийде за межі [0-255]
# застосуйте np.clip(value, 0, 255) та value.astype(np.uint8)
# Виведіть результати обох обробок на екран

import cv2
import numpy as np

img = cv2.imread('darken.png', cv2.IMREAD_COLOR)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ==== 1. ВИРІВНЮВАННЯ ГІСТОГРАМИ ====
value = hsv[:, :, 2]
new_value = cv2.equalizeHist(value)
hsv_eq = hsv.copy()
hsv_eq[:, :, 2] = new_value

img_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

# ==== 2. ЗБІЛЬШЕННЯ ЯСКРАВОСТІ НА 45% ====
value_float = value.astype(np.float32)
value_boost = value_float * 1.45

value_boost = np.clip(value_boost, 0, 255)
value_boost = value_boost.astype(np.uint8)

hsv_boost = hsv.copy()
hsv_boost[:, :, 2] = value_boost

img_boost = cv2.cvtColor(hsv_boost, cv2.COLOR_HSV2BGR)

# ==== ВИВІД ====
cv2.imshow('Original', img)
cv2.imshow('Equalized Value (HSV)', img_eq)
cv2.imshow('Boosted Value (HSV)', img_boost)

cv2.waitKey(0)