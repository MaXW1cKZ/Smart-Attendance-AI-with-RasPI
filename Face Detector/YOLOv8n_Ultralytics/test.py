import cv2
import time
import os
import torch
from ultralytics import YOLO

# --- Configuration ---
USE_WEBCAM = True
VIDEO_FOLDER = '../../Test_Videos/' 
VIDEO_FILENAME = 'video_easy.mp4'
VIDEO_PATH = os.path.join(VIDEO_FOLDER, VIDEO_FILENAME)

YOLO_MODEL_PATH = 'yolov8n_100e.pt'

def main():
    # --- 1. Load Model (YOLOv8n from Ultralytics) ---
    print(f"Running on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # โหลดโมเดล YOLOv8 nano ('yolov8n.pt'). 
    # ** หมายเหตุ: ในโค้ดของคุณใช้ 'yolov8n_100e.pt' **
    # หากคุณมีไฟล์ .pt ที่เทรนมาเฉพาะ ให้ใช้ชื่อนั้น
    # หากต้องการโมเดลทั่วไป ให้ใช้ 'yolov8n-face.pt' (ถ้ามี) หรือ 'yolov8n.pt'
    model = YOLO(YOLO_MODEL_PATH) # < ใช้ชื่อไฟล์โมเดลของคุณ

    # --- Setup Video Capture and Try-Finally block ---
    cap = None
    try:
        if USE_WEBCAM:
            source = 0 # อาจจะต้องเปลี่ยนเป็น 0 ถ้ากล้องไม่ติด
        else:
            source = VIDEO_PATH
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source: {source}")
            return

        frame_count, total_inference_time = 0, 0
        print("Starting detection... Press 'q' to quit.")

        # --- 3. Main Loop ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # --- 4. Detection ---
            start_time = time.time()
            
            # ใช้เมธอด .predict() ของ YOLOv8
            # classes=0 คือการกรองให้หาเฉพาะ class 'person' (ถ้าโมเดลของคุณเทรนมาเป็น 'face' อาจจะต้องเปลี่ยน)
            # verbose=False เพื่อไม่ให้ print log ทุกเฟรม
            results = model.predict(source=frame, classes=0, verbose=False)
            
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            frame_count += 1
            
            # --- 5. Draw Bounding Boxes ---
            # results เป็น list ที่มี 1 item, เราจึงใช้ results[0]
            result = results[0]
            
            # วนลูป Bounding box ที่หาเจอใน result นี้
            for box in result.boxes:
                # กรองด้วยค่า confidence
                if box.conf[0] > 0.5:
                    # ดึงพิกัด xyxy และแปลงเป็น integer
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(c) for c in coords]
                    
                    # ดึงค่า confidence และ class
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    class_name = model.names[class_id]

                    # วาดกรอบและข้อความ
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # --- 6. Display FPS ---
            fps = 1 / inference_time
            cv2.putText(frame, f'FPS: {fps:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('YOLOv8n Benchmark', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # --- 7. Release Resources & Report Results ---
        print("\nCleaning up...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        for i in range(5): cv2.waitKey(1)
        
        if frame_count > 0:
            avg_inference_time_ms = (total_inference_time / frame_count) * 1000
            avg_fps = frame_count / total_inference_time
            print("\n--- Benchmark Results for YOLOv8n ---")
            print(f"Source: {'Webcam' if USE_WEBCAM else VIDEO_FILENAME}")
            print(f"Total Frames Processed: {frame_count}")
            print(f"Average Inference Time: {avg_inference_time_ms:.2f} ms")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Model: {model.ckpt_path}")
            if os.path.exists(YOLO_MODEL_PATH):
                model_size_mb = os.path.getsize(YOLO_MODEL_PATH) / (1024*1024)
                print(f"Model Size: {model_size_mb:.2f} MB ({YOLO_MODEL_PATH})")
            else:
                print(f"Model Size: {YOLO_MODEL_PATH} (Not Found)")
        else:
            print("No frames were processed.")


if __name__ == '__main__':
    main()