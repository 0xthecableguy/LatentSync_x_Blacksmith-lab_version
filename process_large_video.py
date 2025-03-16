import os
import subprocess
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--segment_length", type=int, default=60, help="Length of each segment in seconds")
    args = parser.parse_args()

    # Очистка и создание временной директории
    temp_dir = "temp_segments"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Разделяем видео на сегменты - использую просто число секунд, не формат времени
    cmd = f"ffmpeg -i {args.video_path} -c copy -map 0 -segment_time {args.segment_length} -f segment -reset_timestamps 1 {temp_dir}/segment_%03d.mp4"
    subprocess.run(cmd, shell=True)

    # Разделяем аудио на соответствующие сегменты - также использую просто число секунд
    audio_cmd = f"ffmpeg -i {args.audio_path} -f segment -segment_time {args.segment_length} -c copy {temp_dir}/audio_%03d.mp3"
    subprocess.run(audio_cmd, shell=True)

    # Обрабатываем каждый сегмент
    segment_files = sorted([f for f in os.listdir(temp_dir) if f.startswith("segment_") and f.endswith(".mp4")])
    output_files = []

    for i, segment in enumerate(segment_files):
        segment_path = os.path.join(temp_dir, segment)
        audio_segment = f"audio_{i:03d}.mp3"
        audio_segment_path = os.path.join(temp_dir, audio_segment)
        output_path = os.path.join(temp_dir, f"output_{segment}")
        output_files.append(output_path)

        # Проверяем, существует ли аудиосегмент
        if not os.path.exists(audio_segment_path):
            print(f"Warning: Audio segment {audio_segment_path} not found. Using full audio instead.")
            audio_segment_path = args.audio_path

        process_cmd = f"python -m scripts.inference --unet_config_path 'configs/unet/second_stage.yaml' --inference_ckpt_path 'checkpoints/latentsync_unet.pt' --inference_steps 30 --guidance_scale 1.5 --video_path '{segment_path}' --audio_path '{audio_segment_path}' --video_out_path '{output_path}'"
        subprocess.run(process_cmd, shell=True)

    # Создаем список файлов для объединения с правильным форматом
    with open(f"{temp_dir}/files.txt", "w") as f:
        for output_file in output_files:
            # Используем абсолютные пути, чтобы избежать проблем с относительными путями
            abs_path = os.path.abspath(output_file)
            f.write(f"file '{abs_path}'\n")

    # Объединяем результаты
    merge_cmd = f"ffmpeg -f concat -safe 0 -i {temp_dir}/files.txt -c copy {args.output_path}"
    subprocess.run(merge_cmd, shell=True)

    print(f"Processing complete. Final output: {args.output_path}")


if __name__ == "__main__":
    main()