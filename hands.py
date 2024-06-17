import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model2/keras_model.h5", "Model2/labels.txt")

offset = 20
imgSize = 300

labelss = ["A", "B", "C", "D", "E", "F", "G", "H", "hi", "I",
           "i love u", "J", "K", "L", "M", "N", "O", "P",
           "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

gesture_detected = False
gesture_text = ""
gesture_texts = []
gesture_accumulated_text = ""
gesture_start_time = None
gesture_delay = 5  # Задержка перед очередным распознаванием жеста (в секундах)
initial_delay = True  # Флаг для задержки перед первым распознаванием

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands and len(hands) > 0:  # Проверка наличия рук в кадре
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Проверка на корректность координат
        if x - offset < 0:
            x = offset
        if y - offset < 0:
            y = offset
        if x + w + offset > img.shape[1]:
            w = img.shape[1] - x - offset
        if y + h + offset > img.shape[0]:
            h = img.shape[0] - y - offset

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Отображение текста на экране после распознавания жеста
        if prediction != "None":
            if initial_delay:
                # Задержка перед первым распознаванием
                time.sleep(3)
                initial_delay = False

            if not gesture_detected:
                gesture_detected = True
                gesture_text = labelss[index]
                gesture_texts.append(gesture_text)  # Сохраняем текст в список
                if len(gesture_accumulated_text) > 0:
                    gesture_accumulated_text += " " + gesture_text
                else:
                    gesture_accumulated_text += gesture_text
                gesture_start_time = time.time()

        if gesture_detected:
            # Выводим накопленный текст на экран
            cv2.putText(imgOutput, gesture_accumulated_text, (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

            # Проверяем, прошло ли достаточно времени после распознавания жеста
            if time.time() - gesture_start_time >= gesture_delay:
                gesture_detected = False
                gesture_text = ""

        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

    else:
        # Очищаем текст, если руки не обнаружены
        gesture_detected = False
        gesture_text = ""
        gesture_accumulated_text = ""

    cv2.imshow("Image", imgOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Вывод сохраненных текстов
print("Сохраненные тексты:")
for i, text in enumerate(gesture_texts, start=1):
    print(f"{i}. {text}")
