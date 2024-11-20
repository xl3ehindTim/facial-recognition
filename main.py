import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = [] 

# Initialize some variables
detected_face_locations = []
face_encodings = []
face_names = [] # NOTE: Purely for this example

while True:
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    detected_face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=1, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, detected_face_locations, model="large")

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

        if True in matches:
            # Get the index of the first True match
            first_match_index = matches.index(True)

            # Retrieve the matched face encoding from the known_face_encodings list
            matched_face_encoding = known_face_encodings[first_match_index]
            face_names.append(known_face_names[first_match_index])
        else:
            known_face_encodings.append(face_encoding)
            known_face_names.append(input("What's your name?"))
    
    for (top, right, bottom, left), name in zip(detected_face_locations, face_names):
        # top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
