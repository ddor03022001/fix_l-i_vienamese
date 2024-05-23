import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Đường dẫn tới thư mục chứa mô hình đã tải
model_directory = r"C:\Users\pc\Desktop\fix_loi_vietnamese\model\vietnamese-fix-v2"

# Tải tokenizer và model từ thư mục
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForSeq2SeqLM.from_pretrained(model_directory)

# Tạo pipeline sử dụng mô hình đã tải
corrector = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Định nghĩa hàm xử lý văn bản
def correct_text(text):
    MAX_LENGTH = 512
    prediction = corrector(text, max_length=MAX_LENGTH)[0]
    return prediction['generated_text']

# Tạo giao diện Gradio
iface = gr.Interface(
    fn=correct_text, 
    inputs=gr.Textbox(lines=5, placeholder="Nhập văn bản tiếng Việt cần sửa..."), 
    outputs=gr.Textbox(), 
    title="Sửa lỗi tiếng việt",
    description="Nhập văn bản tiếng Việt cần sửa lỗi chính tả và ngữ pháp."
)

# Chạy Gradio interface
iface.launch()
