import cv2
import time
import os
import numpy as np

# --- Configuration ---
USE_WEBCAM = True
VIDEO_FOLDER = '../../Test_Videos/'
VIDEO_FILENAME = 'video_easy.mp4'
VIDEO_PATH = os.path.join(VIDEO_FOLDER, VIDEO_FILENAME)

# --- YuNet Model Configuration ---
# Path ของไฟล์โมเดล .onnx
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"

def main():
    # --- 1. Load Model (YuNet from OpenCV DNN) ---
    print("Loading YuNet model...")

    # โหลดโมเดล YuNet เข้าสู่ OpenCV's DNN module
    detector = cv2.FaceDetectorYN.create(
        model=YUNET_MODEL_PATH,
        config="",
        input_size=(320, 320), # ขนาด Input ที่โมเดลต้องการ
        score_threshold=0.9,  # กรองผลลัพธ์ที่มั่นใจ > 90%
        nms_threshold=0.3,    # Non-Max Suppression
        top_k=5000
    )
    print("YuNet model loaded successfully")

    # --- Setup Video Capture ---
    cap = None
    try:
        if USE_WEBCAM: source = 0
        else: source = VIDEO_PATH
        cap = cv2.VideoCapture(source)
        if not cap.isOpened(): print(f"Error opening source"); return

        frame_count, total_inference_time = 0, 0
        print("Starting detection... Press 'q' to quit.")

        # --- Main Loop ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_height, frame_width, _ = frame.shape
            # ปรับขนาด Input size ของ detector ให้ตรงกับขนาดเฟรม
            detector.setInputSize((frame_width, frame_height))

            # --- 4. Detection ---
            start_time = time.time()
            # .detect() จะคืนค่า tuple ที่มี results
            _, faces = detector.detect(frame)
            inference_time = time.time() - start_time
            
            total_inference_time += inference_time
            frame_count += 1
            
            # --- 5. Draw Bounding Boxes ---
            if faces is not None:
                for face in faces:
                    # 'face' คือ array ที่มีข้อมูล box, landmarks, confidence
                    box = face[0:4].astype(np.int32)
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    confidence = face[-1]
                    cv2.putText(frame, f'{confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # --- 6. Display FPS ---
            fps = 1 / inference_time
            cv2.putText(frame, f'FPS: {fps:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('YuNet (OpenCV) Benchmark', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        # --- 7. Release & Report ---
        print("\nCleaning up...")
        if cap: cap.release()
        cv2.destroyAllWindows(); [cv2.waitKey(1) for i in range(5)]
        
        if frame_count > 0:
            avg_inference_time_ms = (total_inference_time / frame_count) * 1000
            avg_fps = frame_count / total_inference_time
            print("\n--- Benchmark Results for YuNet ---")
            print(f"Source: {'Webcam' if USE_WEBCAM else VIDEO_FILENAME}")
            print(f"Total Frames Processed: {frame_count}")
            print(f"Average Inference Time: {avg_inference_time_ms:.2f} ms")
            print(f"Average FPS: {avg_fps:.2f}")

            # วัดขนาดไฟล์ .onnx ที่ดาวน์โหลดมา
            if os.path.exists(YUNET_MODEL_PATH):
                model_size_mb = os.path.getsize(YUNET_MODEL_PATH) / (1024*1024)
                print(f"Model Size: {model_size_mb:.2f} MB")
        else:
            print("No frames were processed.")

if __name__ == '__main__':
    main()