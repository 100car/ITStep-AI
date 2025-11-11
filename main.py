import numpy as np

# Завдання 1
# Створіть масив:
#
# nums = np.array([[1, 2, 3, 4],
#                 [5, 6, 7, 8],
#                 [9, 10, 11, 12],
#                 [13, 14, 15 ,16]])
#
# # Використовуючи індекси виведіть:
# # ● число 14
# print("\n", nums[3, 1])
#
# # ● третій рядок
# print("\n", nums[2, :])
#
# # ● перший стовпчик
# print("\n", nums[:, 0])
#
# # ● верхню половину
# print("\n", nums[:2, :])
#
# # ● замініть числа в рядках 2-3 на 100
# nums[1:3, :] = 100
# print("\n", nums)
#
# # ● зробіть другий рядок таким як останній рядок
# nums[1, :] = nums[3, :]
# print("\n", nums)
#

# Завдання 2
# У масиві з попереднього завдання
# nums = np.array([[1, 2, 3, 4],
#                 [5, 6, 7, 8],
#                 [9, 10, 11, 12],
#                 [13, 14, 15 ,16]])
#
# # створіть маску для парних чисел.
# mask = nums % 2 == 0
#
# # З її допомогою
# # ● виведіть самі числа
# print(nums[mask])
# print()
#
# # ● замініть їх на 100
# nums[mask] = 100
# print(nums)

# Завдання 3
# Створіть 2 масиви типу uint8:
# Масив 1: 128 200 10
# Масив 2: 250 10 34
# Об’єднайте їх у пропорції 20% першого масив + 80% другого масиву.
# В результаті має бути тип даних uint8 та числа в діапазоні 0-255

array_1 = np.array([128, 200, 100], dtype=np.uint8)
array_2 = np.array([250, 10, 34], dtype=np.uint8)

result_1 = np.clip(array_1.astype(np.uint16) * 0.2 + array_2.astype(np.uint16) * 0.8, 0, 255).astype(np.uint8)
print(result_1)

result = (array_1.astype(np.uint16) * 0.2 + array_2.astype(np.uint16) * 0.8)
mask = result > 255
result[mask] = 255
result = result.astype(np.uint8)
print(result)

print(result == result_1)

