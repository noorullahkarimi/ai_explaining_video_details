#4th page
def process_video_and_generate_speech(video_path):
    frame_descriptions = process_video(video_path)
    frame_descriptions_text = "\\n".join([f"{v}," for k, v in frame_descriptions.items()])
    summary = summarize_text(frame_descriptions_text)
    spoken_summary = speak(summary)
    return 16000, spoken_summary