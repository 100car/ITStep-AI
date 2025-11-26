# Завдання 1
# Відкрийте зображення data/lesson3/sonet.png. Проведіть бінарізацію.
# Обов’язково використайте:
#  розмиття або наведення різкості
#  адаптивну бінарізацію
#  очищення шумів

import cv2
import numpy as np

# ==== 1. Читання зображення ====
img = cv2.imread("sonet.png")
cv2.imshow('original', img)

# ==== 2. Переведення в відтінки сірого ====
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

# ==== 3. Розмиття для приглушення шумів перед бінаризацією ====
blur = cv2.GaussianBlur(gray, (3, 3), 0)
cv2.imshow("Blur", blur)

# ==== 4. Адаптивна бінаризація ====
th = cv2.adaptiveThreshold(
    blur,                  # розмите ч/б зображення
    255,                   # значення білого
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # адаптивний метод
    cv2.THRESH_BINARY,     # тип бінаризації
    11,                    # розмір сусіднього вікна
    2                      # константа віднімається
)
cv2.imshow("Adaptive Threshold", th)

# ==== 5. Очищення шумів за допомогою bilateralFilter ====
clean = cv2.bilateralFilter(th, 9, 75, 75)
cv2.imshow("Bilateral Clean", clean)

# ==== 6. Завершення ====
cv2.waitKey(0)

