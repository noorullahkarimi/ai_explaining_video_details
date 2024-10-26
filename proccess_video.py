import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
#1st page
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    frame_descriptions = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % fps == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb_frame, return_tensors="pt")
            outputs = model_blip.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            frame_descriptions[frame_count // fps] = caption

        frame_count += 1

    cap.release()
    return frame_descriptions