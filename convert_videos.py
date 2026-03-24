import cv2
import os

# --- PATH TO YOUR FOLDER ---
BASE_PATH = r"C:\Users\Anusha\OneDrive\Desktop\Animal_AI\animals pics"

def convert_all_videos():
    # Loop through every behavior folder (walking, resting, etc.)
    for folder_name in os.listdir(BASE_PATH):
        folder_path = os.path.join(BASE_PATH, folder_name)
        
        if not os.path.isdir(folder_path):
            continue

        print(f"Checking folder: {folder_name}...")
        
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(folder_path, file)
                print(f"  🎥 Converting video: {file}")
                
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) # Frames per second
                count = 0
                success, image = cap.read()
                
                while success:
                    # Save one frame every 1 second (to avoid 1000s of identical images)
                    if count % int(fps) == 0:
                        img_name = f"frame_{file}_{count}.jpg"
                        img_save_path = os.path.join(folder_path, img_name)
                        cv2.imwrite(img_save_path, image)
                    
                    success, image = cap.read()
                    count += 1
                
                cap.release()
                print(f"  ✅ Done! Created images for {file}")

if __name__ == "__main__":
    convert_all_videos()