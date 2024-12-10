#
# Copyright 2024 NXP
#
import cv2
import numpy as np

def preprocess_frame(frame, size):
    min_dim = min(frame.shape[:2])
    start_x = (frame.shape[1] - min_dim) // 2
    start_y = (frame.shape[0] - min_dim) // 2
    cropped_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    resized_frame = cv2.resize(cropped_frame, (size, size))
    normalized_frame = resized_frame / 255.0
    return np.expand_dims(normalized_frame, axis=0).astype(np.float32), resized_frame

def main():
    cap = cv2.VideoCapture(0)  # Capture video from camera
    if not cap.isOpened():
        print("Error: Unable to access the camera")
        return

    # Get native resolution from the camera
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up video writer to save output at camera's native resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used for MP4 files
    out = cv2.VideoWriter('test_inference_video.mp4', fourcc, 30.0, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to fetch frame from the camera")
            break

        # Process frame to get both input to the models
        model_input_448, display_frame_448 = preprocess_frame(frame, 448)
        model_input_224, display_frame_224 = preprocess_frame(frame, 224)
        
        # Show frames
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Model Input Frame 448x448', display_frame_448)
        cv2.imshow('Model Input Frame 224x224', display_frame_224)
        
        # Write original frame to video file
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()