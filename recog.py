import face_recognition
import cv2
import joblib

# Function to recognize faces from live camera feed
def recognize_faces_from_camera(model_path):
    # Load the trained classifier
    clf = joblib.load(model_path)
    
    # Start capturing video from the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        return
    
    print("Recognizing faces. Press 'q' to quit.")
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
        # Convert the frame from BGR to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect all faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Extract face encodings from the frame
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Recognize faces and draw bounding boxes with labels
        for i, face_encoding in enumerate(face_encodings):
            # Predict the name for each face
            name = clf.predict([face_encoding])[0]
            
            # Get the location of the face
            top, right, bottom, left = face_locations[i]
            
            # Draw a bounding box around the face using OpenCV
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Add the name label above the bounding box using OpenCV
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame with bounding boxes and labels
        cv2.imshow("Recognized Faces", frame)
        
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set the path to the trained model
    model_path = "face_recognition_model.joblib"
    
    # Call the function to recognize faces from the camera
    recognize_faces_from_camera(model_path)
