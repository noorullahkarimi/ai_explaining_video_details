from transformers import pipeline
#2th page
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
def summarize_text(text, max_length=10, min_length=2):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]["summary_text"]