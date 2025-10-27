import cv2
import mediapipe as mp
import time
import os

# --- Configuration ---
# ตั้งค่าเพื่อเลือกระหว่าง Webcam หรือไฟล์วิดีโอ
USE_WEBCAM = True # << ลองแก้เป็น True เพื่อทดสอบกับ Webcam สดๆ

# Path ไปยังโฟลเดอร์วิดีโอทดสอบ
VIDEO_FOLDER = '../../Test_Videos/' 
VIDEO_FILENAME = 'video_easy.mp4' # << ลองเปลี่ยนเป็น 'video_medium.mp4' หรือ 'video_hard.mp4'
VIDEO_PATH = os.path.join(VIDEO_FOLDER, VIDEO_FILENAME)

def main():
    # --- 1. Load Model (BlazeFace from MediaPipe) ---
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0, 
        min_detection_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # --- 2. Setup Video Capture ---
    if USE_WEBCAM:
        source = 0
        print("Using Webcam...")
    else:
        source = VIDEO_PATH
        print(f"Using Video File: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return

    frame_count = 0
    total_inference_time = 0
    
    print("Starting detection... Press 'q' to quit.")

    # --- 3. Main Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape
        
        # --- 4. Detection ---
        # แปลงสี BGR (OpenCV) เป็น RGB (MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        start_time = time.time()
        results = face_detection.process(frame_rgb)
        inference_time = time.time() - start_time
        
        total_inference_time += inference_time
        frame_count += 1
        
        # --- 5. Draw Bounding Boxes ---
        if results.detections:
            for detection in results.detections:
                # MediaPipe คืนค่า BBox แบบ Normalize ต้องแปลงกลับ
                box = detection.location_data.relative_bounding_box
                x = int(box.xmin * frame_width)
                y = int(box.ymin * frame_height)
                w = int(box.width * frame_width)
                h = int(box.height * frame_height)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # แสดงค่า Confidence Score
                confidence = detection.score[0]
                cv2.putText(frame, f'{confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # --- 6. Display FPS ---
        fps = 1 / inference_time
        cv2.putText(frame, f'FPS: {fps:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('BlazeFace (MediaPipe) Benchmark', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 7. Report Results ---
    cap.release()
    cv2.destroyAllWindows()
    
    if frame_count > 0:
        avg_inference_time_ms = (total_inference_time / frame_count) * 1000
        avg_fps = frame_count / total_inference_time
    
        print("\n--- Benchmark Results for BlazeFace ---")
        print(f"Source: { 'Webcam' if USE_WEBCAM else VIDEO_FILENAME }")
        print(f"Total Frames Processed: {frame_count}")
        print(f"Average Inference Time: {avg_inference_time_ms:.2f} ms")
        print(f"Average FPS: {avg_fps:.2f}")
        print("Model Size: N/A (Built-in, very small)")
    else:
        print("No frames were processed.")

if __name__ == '__main__':
    main()
