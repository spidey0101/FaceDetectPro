import cv2

# Load pre-trained face detection model
face_model = cv2.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")

# Load pre-trained age and gender estimation models
age_model = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
gender_model = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

# Define the list of age and gender labels
# Define the list of age labels representing different age groups
age_labels = ['(0-2)', '(3-6)', '(7-12)', '(13-17)', '(18-24)', '(25-34)', '(35-44)', '(45-54)', '(55-64)', '(65-74)', '(75-100)']
gender_labels = ['Male', 'Female']

# Initialize the webcam or video capture device
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or provide the path to a video file

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (300, 300))

    # Prepare the input image to be fed into the face detection model
    blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (300, 300), [104, 117, 123], False, False)

    # Set the prepared input blob as the input to the face detection network
    face_model.setInput(blob)
    detections = face_model.forward()

    # Loop over the face detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than a minimum threshold
        if confidence > 0.5:  
            # Get the coordinates of the bounding box for the detected face
            box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face ROI (Region of Interest)
            face_roi = frame[startY:endY, startX:endX]

            # Prepare the input image to be fed into the age and gender estimation models
            face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), False)

            # Set the prepared input blob as the input to the age and gender estimation networks
            age_model.setInput(face_blob)
            age_preds = age_model.forward()
            gender_model.setInput(face_blob)
            gender_preds = gender_model.forward()

            # Get the predicted age and gender
            age = age_labels[age_preds[0].argmax()]
            gender = gender_labels[gender_preds[0].argmax()]

            # Display the age and gender information on the frame
            text = f"{gender}, {age}"
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            # Draw the bounding box around the detected face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
