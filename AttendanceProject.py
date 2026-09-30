import os
from datetime import datetime
from pathlib import Path
import urllib.request

import cv2


def get_face_cascade():
    cascade_path = Path(cv2.__file__).resolve().parent / 'data' / 'haarcascade_frontalface_default.xml'
    if not cascade_path.exists():
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
        cascade_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=20) as response, open(cascade_path, 'wb') as file:
            file.write(response.read())
    return cv2.CascadeClassifier(str(cascade_path))


base_dir = Path(__file__).resolve().parent
path = base_dir / 'Images_Attendance'
attendance_file = base_dir / 'Attendance.csv'

if path.exists() and not path.is_dir():
    backup_path = base_dir / 'Images_Attendance_backup.txt'
    if backup_path.exists():
        backup_path.unlink()
    path.rename(backup_path)
    print(f'Moved old file to {backup_path}')

path.mkdir(exist_ok=True)
attendance_file.touch(exist_ok=True)

face_cascade = get_face_cascade()
known_faces = []

for image_file in sorted(path.iterdir()):
    if not image_file.is_file():
        continue
    image = cv2.imread(str(image_file))
    if image is None:
        continue
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        continue

    x, y, w, h = faces[0]
    face_roi = image[y:y + h, x:x + w]
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    known_faces.append({
        'name': image_file.stem.upper(),
        'hist': hist,
    })

print('Known faces loaded:', [entry['name'] for entry in known_faces])


def markAttendance(name):
    with open(attendance_file, 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            if entry:
                nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tString = time_now.strftime('%H:%M:%S')
            dString = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tString},{dString}')


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face_roi = img[y:y + h, x:x + w]
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)

        best_match_name = 'UNKNOWN'
        best_score = 0.0

        for person in known_faces:
            score = cv2.compareHist(person['hist'], hist, cv2.HISTCMP_CORREL)
            if score > best_score:
                best_score = score
                best_match_name = person['name']

        if best_score > 0.45:
            label = best_match_name
            markAttendance(label)
        else:
            label = 'UNKNOWN'

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('webcam', img)
    if cv2.waitKey(10) == 13:
        break

cap.release()
cv2.destroyAllWindows()