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

torch.jit.script = lambda f: f


# FIX 1: Increase GPU duration to allow longer videos to fully process.
# Default is ~60s which only covers ~8 seconds at ~4.7s/frame inference speed.
@spaces.GPU(duration=3600)
def doo(video, mode, progress=gr.Progress()):
    if mode == "Fast":
        remover = Remover(mode="fast")
    else:
        remover = Remover()

    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # FIX 2: Read fps BEFORE releasing the cap
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    tmpname = str(random.randint(111111111, 999999999))
    processed_frames = 0
    start_time = time.time()

    raw_mp4_path = f"{tmpname}_raw.mp4"
    final_mp4_path = f"{tmpname}.mp4"
    writer_green = None

    png_dir = f"frames_{tmpname}"
    os.makedirs(png_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if time.time() - start_time >= 3600 - 10:
            print("GPU Timeout is coming")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).convert("RGB")

        if writer_green is None:
            w, h = img.size
            writer_green = cv2.VideoWriter(
                raw_mp4_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )

        processed_frames += 1
        print(f"Processing frame {processed_frames}/{total_frames}")
        if total_frames > 0:
            progress(
                processed_frames / total_frames,
                desc=f"Processing frame {processed_frames}/{total_frames}",
            )

        # FIX 3: Run remover inference ONLY ONCE per frame (was called twice before).
        # Previously: remover.process(img, type="green") + remover.process(img, type="rgba")
        # That doubled the GPU time per frame (~4.7s -> ~2.4s per frame).
        # Now: get RGBA once, then derive the green composite manually via alpha blending.
        out_rgba = remover.process(img, type="rgba")
        out_rgba_np = np.array(out_rgba)  # (h, w, 4)

        # Handle unexpected 3-channel output
        if out_rgba_np.ndim == 2 or out_rgba_np.shape[2] == 3:
            alpha_fill = np.full(out_rgba_np.shape[:2] + (1,), 255, dtype=np.uint8)
            out_rgba_np = np.concatenate([out_rgba_np, alpha_fill], axis=2)

        # Save RGBA PNG for WebM
        frame_png_path = os.path.join(png_dir, f"frame_{processed_frames:06d}.png")
        Image.fromarray(out_rgba_np, mode="RGBA").save(frame_png_path)

        # Derive green background via alpha blending (no second inference)
        rgb = out_rgba_np[:, :, :3].astype(np.float32)
        alpha_ch = out_rgba_np[:, :, 3:4].astype(np.float32) / 255.0
        green_bg = np.zeros_like(rgb)
        green_bg[:, :, 1] = 255.0  # pure green
        green_composite = (rgb * alpha_ch + green_bg * (1.0 - alpha_ch)).astype(np.uint8)
        out_green_bgr = cv2.cvtColor(green_composite, cv2.COLOR_RGB2BGR)
        writer_green.write(out_green_bgr)

    cap.release()
    if writer_green is not None:
        writer_green.release()

    # FIX 4: Re-encode with h264 + faststart so moov atom is at the front.
    # mp4v (OpenCV) writes moov at the END — browsers/Gradio can only play
    # the first few buffered seconds without it at the start.
    mp4_ok = False
    if os.path.exists(raw_mp4_path):
        reencode_cmd = [
            "ffmpeg", "-y",
            "-i", raw_mp4_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            final_mp4_path,
        ]
        print("Re-encoding MP4:", " ".join(reencode_cmd))
        try:
            subprocess.run(reencode_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            mp4_ok = True
        except subprocess.CalledProcessError as e:
            print("MP4 re-encode failed:", e.stderr.decode())
            os.rename(raw_mp4_path, final_mp4_path)
            mp4_ok = True
        finally:
            if os.path.exists(raw_mp4_path):
                os.remove(raw_mp4_path)

    # Build WebM with transparency from PNG sequence
    webm_path = f"{tmpname}.webm"
    if os.listdir(png_dir):
        input_pattern = os.path.join(png_dir, "frame_%06d.png")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(int(fps)),
            "-i", input_pattern,
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            webm_path,
        ]
        print("Building WebM:", " ".join(ffmpeg_cmd))
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print("WebM ffmpeg failed:", e.stderr.decode())
            webm_path = None
    else:
        webm_path = None

    shutil.rmtree(png_dir, ignore_errors=True)

    return final_mp4_path if mp4_ok else None, webm_path


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
