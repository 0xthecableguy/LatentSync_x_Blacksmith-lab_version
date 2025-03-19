import os
import subprocess
import argparse
import shutil
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--segment_length", type=int, default=60, help="Length of each segment in seconds")
    args = parser.parse_args()

    # Получаем информацию о видео
    fps_cmd = f"ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate {args.video_path}"
    result = subprocess.run(fps_cmd, shell=True, capture_output=True, text=True)
    fps_str = result.stdout.strip()
    if '/' in fps_str:
        num, den = map(int, fps_str.split('/'))
        fps = num / den
    else:
        fps = float(fps_str)

    duration_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {args.video_path}"
    result = subprocess.run(duration_cmd, shell=True, capture_output=True, text=True)
    total_duration = float(result.stdout.strip())
    print(f"Video duration: {total_duration:.2f}s, FPS: {fps}")

    frames_per_segment = int(args.segment_length * fps)
    total_frames = int(total_duration * fps)

    # Создаем временную директорию для всех кадров
    with tempfile.TemporaryDirectory() as all_frames_dir:
        frames_dir = Path(all_frames_dir) / "frames"
        frames_dir.mkdir(exist_ok=True)
        output_frames_dir = Path(all_frames_dir) / "output_frames"
        output_frames_dir.mkdir(exist_ok=True)

        # Извлекаем все кадры из видео
        print("Extracting frames from video...")
        extract_cmd = f"ffmpeg -i {args.video_path} -qscale:v 1 {frames_dir}/%05d.png"
        subprocess.run(extract_cmd, shell=True)

        # Получаем список всех кадров
        all_frames = sorted(list(frames_dir.glob("*.png")))
        if not all_frames:
            print("No frames were extracted!")
            return

        print(f"Extracted {len(all_frames)} frames")

        # Обрабатываем видео посегментно
        for start_frame in range(0, len(all_frames), frames_per_segment):
            end_frame = min(start_frame + frames_per_segment, len(all_frames))
            segment_frames = all_frames[start_frame:end_frame]

            segment_idx = start_frame // frames_per_segment
            print(f"Processing segment {segment_idx + 1}: frames {start_frame + 1}-{end_frame}")

            # Создаем временные директории для этого сегмента
            with tempfile.TemporaryDirectory() as temp_dir:
                segment_frames_dir = Path(temp_dir) / "segment_frames"
                segment_frames_dir.mkdir(exist_ok=True)

                # Создаем символические ссылки на кадры сегмента
                for i, frame in enumerate(segment_frames):
                    dest = segment_frames_dir / f"{i + 1:05d}.png"
                    shutil.copy(frame, dest)

                # Создаем видео из кадров
                segment_video = Path(temp_dir) / "segment.mp4"
                create_video_cmd = f"ffmpeg -framerate {fps} -i {segment_frames_dir}/%05d.png -c:v libx264 -pix_fmt yuv420p {segment_video}"
                subprocess.run(create_video_cmd, shell=True)

                # Определяем временной диапазон для аудио
                start_time = start_frame / fps
                end_time = end_frame / fps

                # Извлекаем аудио для этого сегмента
                segment_audio = Path(temp_dir) / "segment_audio.wav"
                extract_audio_cmd = f"ffmpeg -ss {start_time:.3f} -to {end_time:.3f} -i {args.audio_path} -c:a pcm_s16le {segment_audio}"
                subprocess.run(extract_audio_cmd, shell=True)

                # Обрабатываем сегмент с LatentSync
                output_video = Path(temp_dir) / "output.mp4"
                process_cmd = f"python -m scripts.inference --unet_config_path 'configs/unet/stage2.yaml' --inference_ckpt_path 'checkpoints/latentsync_unet.pt' --inference_steps 50 --guidance_scale 1.5 --video_path '{segment_video}' --audio_path '{segment_audio}' --video_out_path '{output_video}'"
                try:
                    print(f"Running inference on segment...")
                    subprocess.run(process_cmd, shell=True, check=True)
                    print(f"Segment processed successfully")

                    # Извлекаем кадры из обработанного видео
                    extract_output_cmd = f"ffmpeg -i {output_video} -qscale:v 1 {temp_dir}/output_%05d.png"
                    subprocess.run(extract_output_cmd, shell=True)

                    # Копируем обработанные кадры в общую директорию
                    output_frames = sorted(list(Path(temp_dir).glob("output_*.png")))
                    for i, frame in enumerate(output_frames):
                        global_frame_idx = start_frame + i + 1
                        shutil.copy(frame, output_frames_dir / f"{global_frame_idx:05d}.png")

                except Exception as e:
                    print(f"Error processing segment: {e}")
                    # В случае ошибки копируем исходные кадры
                    for i, frame in enumerate(segment_frames):
                        global_frame_idx = start_frame + i + 1
                        shutil.copy(frame, output_frames_dir / f"{global_frame_idx:05d}.png")

        # Получаем финальное видео из всех обработанных кадров
        print("Creating final video from processed frames...")
        video_without_audio = "temp_video_no_audio.mp4"
        create_final_cmd = f"ffmpeg -framerate {fps} -i {output_frames_dir}/%05d.png -c:v libx264 -pix_fmt yuv420p {video_without_audio}"
        subprocess.run(create_final_cmd, shell=True)

        # Добавляем оригинальное аудио
        print("Adding original audio to video...")
        final_cmd = f"ffmpeg -i {video_without_audio} -i {args.audio_path} -map 0:v -map 1:a -c:v copy -c:a aac -shortest {args.output_path}"
        subprocess.run(final_cmd, shell=True)

        # Удаляем временное видео без аудио
        if os.path.exists(video_without_audio):
            os.remove(video_without_audio)

    print(f"Processing complete! Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()