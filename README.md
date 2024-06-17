# Основная информация

![Python-3.10.6](https://img.shields.io/badge/Python-v3.10.6-blue?style=for-the-badge)

---

Для работы исполняемого файла ***hands.py*** потребуются следующие библиотеки: 

```python
import cv2
import cvzone.HandTrackingModule
import cvzone.ClassificationModule
import numpy
import math
import time
```


Файл ***keras_model.h5*** содержит в себе набор классов для каждой буквы алфавита

Файл ***labels.txt*** хранит в себе метки классов, которые используются для отображения результатов предсказаний модели

Фрагмент обучения модели представлен в этом файле + keras_model.py ([keras_model.py](https://github.com/Dante1902/HANDS/blob/main/keras_model.py))
