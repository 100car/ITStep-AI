import cv2
import numpy as np

# Завдання 1
# Відкрийте зображення data/Lenna.png. Прочитайте маски data/mask1.png та data/mask2.png.
# Об’єднайте дві маски в одну, скористайтесь cv2.bitwise_or() та виведіть результат

# img = cv2.imread(
#      'data/lesson1/Lenna.png',
#      cv2.IMREAD_GRAYSCALE
# )
#
# mask_1 = cv2.imread(
#      'data/lesson1/mask1.png',
#      cv2.IMREAD_GRAYSCALE
# )
#
# mask_2 = cv2.imread(
#      'data/lesson1/mask2.png',
#      cv2.IMREAD_GRAYSCALE
# )
#
# cv2.imshow('Lenna', img)
# cv2.imshow('mask_1', mask_1)
# cv2.imshow('mask_2', mask_2)
#
# # Перетворюємо маски в булевий тип
# mask1_bool = mask_1.astype(bool)
# mask2_bool = mask_2.astype(bool)
#
# # Частина зображення, що відповідає mask1
# part_mask1 = img.copy()
# part_mask1[~mask1_bool] = 0   # де маска False — ставимо чорний
#
# # Частина зображення, що відповідає mask2
# part_mask2 = img.copy()
# part_mask2[~mask2_bool] = 0
#
# # Частина зображення, що відповідає одночасно mask1 і mask2
# mask_both = mask1_bool & mask2_bool
# part_both = img.copy()
# part_both[~mask_both] = 0
#
# # Об'єднання двох масок через cv2.bitwise_or
# mask_union = cv2.bitwise_or(mask_1, mask_2)
#
# # Показуємо результати
# cv2.imshow('mask_union', mask_union)
# cv2.imshow('part_mask1', part_mask1)
# cv2.imshow('part_mask2', part_mask2)
# cv2.imshow('part_both', part_both)
#
# cv2.waitKey(0)


# Завдання 2
# Виведіть зображення. Підберіть самостійно межі

baboo = cv2.imread(
     'data/lesson1/baboo.jpg',
     cv2.IMREAD_GRAYSCALE
)

cv2.imshow('baboo', baboo)

eyes_baboo = baboo[5:55, 65:-65]
cv2.imshow('eyes baboo', eyes_baboo)

cv2.waitKey(0)