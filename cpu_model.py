import torch
import cv2
import time
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

cpu_model = model.to('cpu')
quantized_model = torch.quantization.quantize_dynamic(
    cpu_model, {torch.nn.Linear}, dtype=torch.qint8
)

def process_video(video_path, model, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0
    total_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        inputs = processor(images=pil_image, return_tensors="pt").pixel_values.to('cpu')
        outputs = model.generate(inputs)
        text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        end_time = time.time()
        total_time += (end_time - start_time)
        frame_count += 1

        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()

    fps = frame_count / total_time
    return fps


input_video_path = r'E:\tensorgo\videoplayback.mp4'
gpu_output_video_path = 'gpu_output_video.mp4'
cpu_output_video_path = 'cpu_output_video.mp4'

if torch.cuda.is_available():
    gpu_fps = process_video(input_video_path, model, gpu_output_video_path)
    print(f"GPU Model FPS: {gpu_fps}")

cpu_fps = process_video(input_video_path, quantized_model, cpu_output_video_path)
print(f"CPU Model FPS: {cpu_fps}")

print(f"GPU Model FPS: {gpu_fps if torch.cuda.is_available() else 'N/A'}")
print(f"CPU Model FPS: {cpu_fps}")
