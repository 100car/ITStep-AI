import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import numpy as np

# НАЗВИ КЛАСІВ ПОРІД СОБАК
class_names = [
            'beagle',
            'bulldog',
            'dalmatian',
            'german-shepherd',
            'husky',
            'labrador-retriever',
            'poodle',
            'rottweiler']

# 1. відкриваємо модель
session = ort.InferenceSession("model.onnx")
print(session)

# 2. відкриваємо зображення
image = Image.open("husky10.jpg").convert("RGB")

# 3. трансформації зображення
transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 4. застосовуємо трансформації
input_tensor = transformer(image)

# 5. додаємо batch dimension (1, 3, 224, 224)
input_tensor = input_tensor.unsqueeze(0)

# 6. переводимо в numpy (ONNX так хоче)
input_tensor = input_tensor.numpy()

# використання моделі для передбачення
result = session.run(
    None,
    {"input": input_tensor}
)
# 7. виводимо результат
print(np.array(result).shape)
print(result)
result =result[0][0]

# 8. знаходимо клас з найбільшим значенням
ind = result.argmax()
print(f"Індекс найбільшої ймовірності: {ind}")
print(f"Це порода: {class_names[ind]}")
# отримуємо ймовірність передбаченого класу
max_num = result.max()
result -= max_num
exp_result = np.exp(result)
sum_exp = exp_result.sum()
probs = exp_result / sum_exp
prob = probs[ind]
print(f"Ймовірність передбаченого класу: {prob:.4f}")

image.show(f"Input Image - Predicted: {class_names[ind]}, Probability: {prob:.4f}")




#
# predicted_class = np.argmax(result[0], axis=1)
# print(f"Predicted class: {predicted_class[0]}")
#
# # 9. виводимо ймовірності для кожного класу
# probabilities = result[0][0]
# for i, prob in enumerate(probabilities):
#     print(f"Class {i}: Probability {prob:.4f}")
# # 10. виводимо ймовірність для передбаченого класу
# predicted_prob = probabilities[predicted_class[0]]
# print(f"Predicted class probability: {predicted_prob:.4f}")
# # 11. виводимо топ-5 класів з їх ймовірностями
# top5_indices = np.argsort(probabilities)[-5:][::-1]
# print("Top 5 classes:")
# for i in top5_indices:
#     print(f"Class {i}: Probability {probabilities[i]:.4f}")
# # 12. зберігаємо ймовірності в файл
# np.savetxt("probabilities.txt", probabilities)
# print("Probabilities saved to probabilities.txt")
# # 13. зберігаємо топ-5 класів в файл
# with open("top5_classes.txt", "w") as f:
#     for i in top5_indices:
#         f.write(f"Class {i}: Probability {probabilities[i]:.4f}\n")
# print("Top 5 classes saved to top5_classes.txt")
# # 14. зберігаємо передбачений клас в файл
# with open("predicted_class.txt", "w") as f:
#     f.write(f"Predicted class: {predicted_class[0]}\n")
#     f.write(f"Predicted class probability: {predicted_prob:.4f}\n")
# print("Predicted class saved to predicted_class.txt")
# # 15. зберігаємо модель в інший файл (копія)
# import shutil
# shutil.copy("model.onnx", "model_copy.onnx")
# print("Model copied to model_copy.onnx")
# # 16. виводимо інформацію про вхідні та вихідні ш
# print("Input details:")
# for input_meta in session.get_inputs():
#     print(f"Name: {input_meta.name}, Shape: {input_meta.shape}, Type: {input_meta.type}")
# print("Output details:")
# for output_meta in session.get_outputs():
#     print(f"Name: {output_meta.name}, Shape: {output_meta.shape}, Type: {output_meta.type}")
#
#
#
#
#
