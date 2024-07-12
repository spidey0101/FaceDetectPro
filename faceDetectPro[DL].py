from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Load pre-trained face detection model
face_model = cv2.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
age_model = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
gender_model = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

age_labels = ['(0-2)', '(3-6)', '(7-12)', '(13-17)', '(18-24)', '(25-34)', '(35-44)', '(45-54)', '(55-64)', '(65-74)', '(75-100)']
gender_labels = ['Male', 'Female']

def detect_faces_and_attributes(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    face_model.setInput(blob)
    detections = face_model.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            (startX, startY, endX, endY) = box.astype("int")
            face_roi = frame[startY:endY, startX:endX]

            face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), False)
            age_model.setInput(face_blob)
            age_preds = age_model.forward()
            gender_model.setInput(face_blob)
            gender_preds = gender_model.forward()

            age = age_labels[age_preds[0].argmax()]
            gender = gender_labels[gender_preds[0].argmax()]

            text = f"{gender}, {age}"
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_faces_and_attributes(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
