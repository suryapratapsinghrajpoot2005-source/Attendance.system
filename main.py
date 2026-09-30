import cv2
from pathlib import Path
import urllib.request


def get_face_cascade():
    cascade_path = Path(cv2.__file__).resolve().parent / 'data' / 'haarcascade_frontalface_default.xml'
    if not cascade_path.exists():
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
        cascade_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=20) as response, open(cascade_path, 'wb') as file:
            file.write(response.read())
    return cv2.CascadeClassifier(str(cascade_path))


base_dir = Path(__file__).resolve().parent
img_dir = base_dir / 'Images_Attendance'
image_files = sorted(
    [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}],
    key=lambda p: p.name.lower(),
)

first_image = image_files[0] if image_files else None
second_image = image_files[1] if len(image_files) > 1 else first_image

face_cascade = get_face_cascade()


def detect_face_in_image(image_path, window_name=None):
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f'Could not load image: {image_path}')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        raise SystemExit(f'No face found in {image_path.name}')

    x, y, w, h = faces[0]
    if window_name:
        cv2.rectangle(image, (x, y), (x + w, y + h), (155, 0, 255), 2)
        cv2.imshow(window_name, image)
    return image[y:y + h, x:x + w]


def detect_face_in_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face_roi = frame[y:y + h, x:x + w]
    return face_roi, (x, y, w, h)


def face_histogram(face_roi):
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def compare_static_images():
    face1 = detect_face_in_image(first_image, 'Reference Image')
    face2 = detect_face_in_image(second_image, 'Second Image')

    hist1 = face_histogram(face1)
    hist2 = face_histogram(face2)
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    print('Face similarity score:', round(float(similarity), 4))
    print('Faces matched:' if similarity > 0.5 else 'Faces not matched:')

    cv2.putText(face2, f'Similarity: {similarity:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Second Image', face2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_known_faces(image_dir):
    known_faces = {}
    if not image_dir.exists():
        return known_faces

    for image_path in sorted(image_dir.iterdir(), key=lambda p: p.name.lower()):
        if not image_path.is_file() or image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        detected = detect_face_in_frame(image)
        if detected is None:
            continue

        face_roi, _ = detected
        known_faces[image_path.stem] = face_histogram(face_roi)

    return known_faces


def capture_reference_face():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Could not open the camera.')
        raise SystemExit(1)

    print('Look at the camera and press c to capture the reference face, q to quit.')
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to read from camera.')
            cap.release()
            raise SystemExit(1)

        detected = detect_face_in_frame(frame)
        if detected is not None:
            face_roi, (x, y, w, h) = detected
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Press c to capture reference face', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No face detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Reference Face Capture', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if detected is None:
                print('No face detected. Please retry.')
                continue
            cap.release()
            cv2.destroyAllWindows()
            return detected[0]
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit(0)


def find_best_match(frame, known_faces):
    detected = detect_face_in_frame(frame)
    if detected is None:
        return None, 0.0, None

    face_roi, (x, y, w, h) = detected
    live_hist = face_histogram(face_roi)
    best_name = None
    best_score = -1.0

    for name, reference_hist in known_faces.items():
        score = cv2.compareHist(reference_hist, live_hist, cv2.HISTCMP_CORREL)
        if score > best_score:
            best_score = score
            best_name = name

    return best_name, best_score, (x, y, w, h)


def run_camera():
    known_faces = load_known_faces(img_dir)

    if not known_faces:
        print('No saved face images found. Press c to capture a reference face manually.')
        reference_face = capture_reference_face()
        known_faces = {'Captured Person': face_histogram(reference_face)}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Could not open the camera.')
        raise SystemExit(1)

    print('Camera mode: press q to quit.')
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to read from camera.')
            break

        name, score, box = find_best_match(frame, known_faces)
        if box is not None:
            x, y, w, h = box
            if name is not None and score > 0.5:
                color = (0, 255, 0)
                label = f'{name} {score:.2f}'
            else:
                color = (0, 0, 255)
                label = 'Unknown Person'
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, 'No face detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Camera Face Match', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_camera()