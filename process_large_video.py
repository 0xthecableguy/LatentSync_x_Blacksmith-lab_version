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

    # Разделяем видео на сегменты
    print(f"Splitting video into {args.segment_length}-second segments...")
    cmd = f"ffmpeg -i {args.video_path} -c copy -map 0 -segment_time {args.segment_length} -f segment -reset_timestamps 1 {temp_dir}/segment_%03d.mp4"
    subprocess.run(cmd, shell=True)

    # Разделяем аудио на соответствующие сегменты
    print(f"Splitting audio into {args.segment_length}-second segments...")
    audio_cmd = f"ffmpeg -i {args.audio_path} -f segment -segment_time {args.segment_length} -c copy {temp_dir}/audio_%03d.mp3"
    subprocess.run(audio_cmd, shell=True)

    # Обрабатываем каждый сегмент
    segment_files = sorted([f for f in os.listdir(temp_dir) if f.startswith("segment_") and f.endswith(".mp4")])
    output_files = []

    print(f"Found {len(segment_files)} video segments to process")

    for i, segment in enumerate(segment_files):
        print(f"\nProcessing segment {i + 1}/{len(segment_files)}: {segment}")
        segment_path = os.path.join(temp_dir, segment)
        audio_segment = f"audio_{i:03d}.mp3"
        audio_segment_path = os.path.join(temp_dir, audio_segment)
        output_path = os.path.join(temp_dir, f"output_{i:03d}.mp4")
        output_files.append(output_path)

        # Проверяем, существует ли аудиосегмент
        if not os.path.exists(audio_segment_path):
            print(f"Warning: Audio segment {audio_segment_path} not found. Using full audio instead.")
            audio_segment_path = args.audio_path

        process_cmd = f"python -m scripts.inference --unet_config_path 'configs/unet/stage2.yaml' --inference_ckpt_path 'checkpoints/latentsync_unet.pt' --inference_steps 30 --guidance_scale 1.5 --video_path '{segment_path}' --audio_path '{audio_segment_path}' --video_out_path '{output_path}'"
        try:
            print(f"Running inference on segment...")
            subprocess.run(process_cmd, shell=True, check=True)
            print(f"Segment processed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error processing segment: {e}")
            continue

        # Проверяем, создался ли выходной файл
        if not os.path.exists(output_path):
            print(f"Warning: Output file {output_path} was not created")
            continue

    # Проверяем, есть ли обработанные файлы
    processed_files = [f for f in output_files if os.path.exists(f)]
    if not processed_files:
        print("Error: No segments were processed")
        shutil.rmtree(temp_dir)
        return

    # Создаем список файлов для объединения с правильным форматом
    with open(f"{temp_dir}/files.txt", "w") as f:
        for output_file in processed_files:
            # Используем абсолютные пути, чтобы избежать проблем с относительными путями
            abs_path = os.path.abspath(output_file)
            f.write(f"file '{abs_path}'\n")

    # Объединяем результаты
    print("\nMerging processed segments...")
    merge_cmd = f"ffmpeg -f concat -safe 0 -i {temp_dir}/files.txt -c copy {args.output_path}"
    subprocess.run(merge_cmd, shell=True)

    print(f"\nProcessing complete. Final output: {args.output_path}")

    # Очистка временных файлов
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()