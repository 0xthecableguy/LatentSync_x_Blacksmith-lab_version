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
    parser.add_argument("--overlap", type=int, default=5, help="Overlap between segments in seconds")
    args = parser.parse_args()

    # Определяем расширение аудио файла для сохранения того же формата
    audio_ext = os.path.splitext(args.audio_path)[1]  # Получаем расширение (.mp3, .wav, и т.д.)

    # Получаем информацию о длительности видео
    duration_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {args.video_path}"
    result = subprocess.run(duration_cmd, shell=True, capture_output=True, text=True)
    total_duration = float(result.stdout.strip())
    print(f"Total video duration: {total_duration:.2f} seconds")

    # Очистка и создание временной директории
    temp_dir = "temp_segments"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Создаем сегменты с перекрытием
    segment_times = []
    start_time = 0

    while start_time < total_duration:
        end_time = min(start_time + args.segment_length, total_duration)
        segment_times.append((start_time, end_time))
        start_time = end_time - args.overlap  # Вычитаем перекрытие

        # Предотвращаем слишком короткие последние сегменты
        if start_time >= total_duration - args.overlap:
            break

    print(f"Will create {len(segment_times)} segments with {args.overlap}s overlap")

    # Обрабатываем каждый сегмент
    processed_files = []
    intermediate_files = []

    for i, (start, end) in enumerate(segment_times):
        print(f"\nProcessing segment {i + 1}/{len(segment_times)}: {start:.2f}s - {end:.2f}s")

        segment_path = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
        intermediate_files.append(segment_path)

        audio_segment_path = os.path.join(temp_dir, f"audio_{i:03d}{audio_ext}")
        intermediate_files.append(audio_segment_path)

        output_path = os.path.join(temp_dir, f"output_{i:03d}.mp4")
        processed_files.append(output_path)

        # Вырезаем сегмент видео с точными временными метками
        print(f"Extracting video segment from {start:.2f}s to {end:.2f}s...")
        video_cmd = f"ffmpeg -ss {start:.3f} -to {end:.3f} -i {args.video_path} -c:v libx264 -preset ultrafast -force_key_frames 0 -avoid_negative_ts 1 {segment_path}"
        subprocess.run(video_cmd, shell=True)

        # Вырезаем соответствующий сегмент аудио, сохраняя оригинальный кодек
        print(f"Extracting audio segment...")
        audio_cmd = f"ffmpeg -ss {start:.3f} -to {end:.3f} -i {args.audio_path} -c:a copy {audio_segment_path}"
        subprocess.run(audio_cmd, shell=True)

        # Обрабатываем сегмент с помощью LatentSync
        process_cmd = f"python -m scripts.inference --unet_config_path 'configs/unet/stage2.yaml' --inference_ckpt_path 'checkpoints/latentsync_unet.pt' --inference_steps 50 --guidance_scale 1.5 --video_path '{segment_path}' --audio_path '{audio_segment_path}' --video_out_path '{output_path}'"
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

    # Убеждаемся, что у нас есть обработанные файлы
    existing_processed_files = [f for f in processed_files if os.path.exists(f)]
    if not existing_processed_files:
        print("Error: No segments were processed")
        shutil.rmtree(temp_dir)
        return

    # Создаем список файлов для объединения
    concat_file = os.path.join(temp_dir, "files.txt")
    with open(concat_file, "w") as f:
        for file_path in existing_processed_files:
            # Используем относительные пути
            rel_path = os.path.basename(file_path)
            f.write(f"file '{rel_path}'\n")

    # Объединяем видео сегменты без аудио
    print("\nMerging processed video segments without audio...")
    temp_video_without_audio = os.path.join(temp_dir, "temp_video_no_audio.mp4")
    merge_cmd = f"cd {temp_dir} && ffmpeg -f concat -safe 0 -i files.txt -c:v copy -an {os.path.basename(temp_video_without_audio)}"
    subprocess.run(merge_cmd, shell=True)

    # Если видео без аудио не создалось, завершаем с ошибкой
    if not os.path.exists(temp_video_without_audio):
        print("Error: Failed to create merged video without audio")
        shutil.rmtree(temp_dir)
        return

    # Добавляем полную оригинальную аудиодорожку к финальному видео
    print("\nAdding original audio to the final video...")
    final_cmd = f"ffmpeg -i {temp_video_without_audio} -i {args.audio_path} -map 0:v -map 1:a -c:v copy -c:a aac -shortest {args.output_path}"
    subprocess.run(final_cmd, shell=True)

    print(f"\nProcessing complete. Final output: {args.output_path}")

    # Очистка временных файлов
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()