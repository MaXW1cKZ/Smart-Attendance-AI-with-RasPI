import cv2
import time
import os
from facenet_pytorch import MTCNN
import torch

# --- Configuration ---
USE_WEBCAM = True 
VIDEO_FOLDER = '../../Test_Videos/' 
VIDEO_FILENAME = 'video_easy.mp4'
VIDEO_PATH = os.path.join(VIDEO_FOLDER, VIDEO_FILENAME)

def main():
    # --- 1. Load Model (MTCNN from facenet-pytorch) ---
    # ตรวจสอบว่ามี GPU หรือไม่, ถ้าไม่มีให้ใช้ CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    
    # keep_all=True เพื่อหาทุกใบหน้าในภาพ, post_process=False เพื่อให้ทำงานเร็วขึ้น (เราไม่ต้องการ normalize ภาพ)
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False)

    # --- 2. Setup Video Capture --- (เหมือนเดิม)
    if USE_WEBCAM:
        source = 0
    else:
        source = VIDEO_PATH
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): print(f"Error opening video source"); return

    frame_count, total_inference_time = 0, 0
    print("Starting detection... Press 'q' to quit.")

    # --- 3. Main Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- 4. Detection ---
        # MTCNN ไม่ต้องการการแปลงสี RGB, ทำงานกับ BGR ได้ (แต่ภายใน library อาจมีการแปลง)
        start_time = time.time()
        # .detect() จะคืนค่า list ของ bounding boxes และ list ของ probabilities
        boxes, probs = mtcnn.detect(frame)
        inference_time = time.time() - start_time
        
        total_inference_time += inference_time
        frame_count += 1
        
        # --- 5. Draw Bounding Boxes ---
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob > 0.9: # กรองเอาเฉพาะที่ความมั่นใจสูง
                    # box คือ [x1, y1, x2, y2]
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'{prob:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # --- 6. Display FPS --- (เหมือนเดิม)
        fps = 1 / inference_time
        cv2.putText(frame, f'FPS: {fps:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('MTCNN Benchmark', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # --- 7. Report Results --- (เหมือนเดิม)
    cap.release(); cv2.destroyAllWindows()
    
    if frame_count > 0:
        avg_inference_time_ms = (total_inference_time / frame_count) * 1000
        avg_fps = frame_count / total_inference_time

        print("\n--- Benchmark Results for MTCNN ---")
        print(f"Source: {'Webcam' if USE_WEBCAM else VIDEO_FILENAME}")
        print(f"Total Frames Processed: {frame_count}")
        print(f"Average Inference Time: {avg_inference_time_ms:.2f} ms")
        print(f"Average FPS: {avg_fps:.2f}")
        print("Model Size: ~ (Installed via library, check cache for .pth files if needed)")
    else:
        print("No frames were processed.")

if __name__ == '__main__':
    main()
