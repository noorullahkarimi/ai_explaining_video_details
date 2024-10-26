import gradio as gr
#5th page
with gr.Blocks() as interface:
    gr.Markdown("# مولد کپشن برای فریم‌های ویدیو")
    with gr.Row():
        video_input = gr.Video(label="آپلود فایل ویدیو")
        audio_output = gr.Audio(label="خروجی گفتار", type="numpy")

    video_input.upload(process_video_and_generate_speech, inputs=video_input, outputs=audio_output)

interface.launch()