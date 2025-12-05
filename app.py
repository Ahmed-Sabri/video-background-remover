import os
import shutil
import subprocess
import spaces
import gradio as gr
import cv2
import numpy as np
import time
import random
from PIL import Image
import torch

from transparent_background import Remover

# Disable TorchScript to avoid issues with some models
torch.jit.script = lambda f: f


@spaces.GPU()
def doo(video, mode, progress=gr.Progress()):
    # Choose remover mode
    if mode == "Fast":
        remover = Remover(mode="fast")
    else:
        remover = Remover()

    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tmpname = str(random.randint(111111111, 999999999))
    processed_frames = 0
    start_time = time.time()

    writer_green = None  # MP4 with green background

    # Temp directory for RGBA PNG frames
    png_dir = f"frames_{tmpname}"
    os.makedirs(png_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Timeout safeguard (~20 minutes)
        if time.time() - start_time >= 20 * 60 - 5:
            print("GPU Timeout is coming")
            break

        # OpenCV BGR -> PIL RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).convert("RGB")

        # Initialize writer lazily once we know size
        if writer_green is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w, h = img.size

            writer_green = cv2.VideoWriter(
                f"{tmpname}.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )

        processed_frames += 1
        print(f"Processing frame {processed_frames}")
        if total_frames > 0:
            progress(
                processed_frames / total_frames,
                desc=f"Processing frame {processed_frames}/{total_frames}",
            )

        # 1) Green background version (RGB)
        out_green = remover.process(img, type="green")
        out_green_bgr = cv2.cvtColor(np.array(out_green), cv2.COLOR_RGB2BGR)
        writer_green.write(out_green_bgr)

        # 2) Transparent version (RGBA) -> save as PNG
        out_rgba = remover.process(img, type="rgba")
        out_rgba_np = np.array(out_rgba)  # (h, w, 4)
        # Ensure 4 channels
        if out_rgba_np.shape[2] == 3:
            # If library unexpectedly returns RGB, add an opaque alpha
            alpha = np.full(out_rgba_np.shape[:2] + (1,), 255, dtype=np.uint8)
            out_rgba_np = np.concatenate([out_rgba_np, alpha], axis=2)

        frame_png_path = os.path.join(png_dir, f"frame_{processed_frames:06d}.png")
        Image.fromarray(out_rgba_np, mode="RGBA").save(frame_png_path)

    cap.release()
    if writer_green is not None:
        writer_green.release()

    # Build WebM with alpha from PNG sequence using ffmpeg
    webm_path = f"{tmpname}.webm"
    if os.listdir(png_dir):
        # Example ffmpeg command:
        # ffmpeg -y -framerate <fps> -i frames_%06d.png -c:v libvpx-vp9 -pix_fmt yuva420p out.webm
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        input_pattern = os.path.join(png_dir, "frame_%06d.png")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(int(fps)),
            "-i",
            input_pattern,
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuva420p",
            webm_path,
        ]
        print("Running ffmpeg:", " ".join(ffmpeg_cmd))
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print("ffmpeg failed:", e)
            # If ffmpeg fails, you still have the MP4; just skip WebM
            webm_path = None
    else:
        webm_path = None

    # Cleanup PNG frames
    shutil.rmtree(png_dir, ignore_errors=True)

    # Return both outputs; if WebM failed, return None for it
    return f"{tmpname}.mp4", webm_path


examples = [["./mp4.mp4"]]

iface = gr.Interface(
    fn=doo,
    inputs=[
        "video",
        gr.Radio(
            ["Normal", "Fast"],
            label="Select mode",
            value="Normal",
            info=(
                "Normal is more accurate, but takes longer. | "
                "Fast has lower accuracy so the process will be faster."
            ),
        ),
    ],
    outputs=[
        gr.Video(label="Green Screen MP4"),
        gr.Video(label="Transparent WebM"),
    ],
    examples=examples,
)

if __name__ == "__main__":
    iface.launch()
