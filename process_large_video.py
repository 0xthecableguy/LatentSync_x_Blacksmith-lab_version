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
    parser.add_argument("--overlap", type=int, default=5, help="Overlap for segments in seconds")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary files after processing")
    args = parser.parse_args()

    # Получаем информацию о длительности видео
    duration_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {args.video_path}"
    result = subprocess.run(duration_cmd, shell=True, capture_output=True, text=True)
    total_duration = float(result.stdout.strip())
    print(f"Total video duration: {total_duration:.2f} seconds")

    # Очистка и создание временных директорий
    temp_dir = "temp_segments"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    segments_dir = os.path.join(temp_dir, "segments")
    audio_segments_dir = os.path.join(temp_dir, "audio_segments")
    processed_dir = os.path.join(temp_dir, "processed")
    trimmed_dir = os.path.join(temp_dir, "trimmed")

    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(audio_segments_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(trimmed_dir, exist_ok=True)

    # Создаем сегменты с перекрытием
    segment_times = []
    start_time = 0

    while start_time < total_duration:
        end_time = min(start_time + args.segment_length + args.overlap, total_duration)
        segment_times.append((start_time, end_time))
        start_time += args.segment_length  # Следующий сегмент начинается через segment_length

    print(f"Will create {len(segment_times)} segments with {args.overlap}s overlap")

    # Нарезаем сегменты и обрабатываем их
    processed_files = []

    for i, (start, end) in enumerate(segment_times):
        print(f"\nProcessing segment {i + 1}/{len(segment_times)}: {start:.2f}s - {end:.2f}s")

        segment_path = os.path.join(segments_dir, f"segment_{i:03d}.mp4")
        audio_segment_path = os.path.join(audio_segments_dir, f"audio_{i:03d}.wav")
        output_path = os.path.join(processed_dir, f"processed_{i:03d}.mp4")
        trimmed_path = os.path.join(trimmed_dir, f"trimmed_{i:03d}.mp4")

        # Вырезаем сегмент видео
        print(f"Extracting video segment from {start:.2f}s to {end:.2f}s...")
        video_cmd = f"ffmpeg -ss {start:.3f} -to {end:.3f} -i {args.video_path} -c:v libx264 -preset ultrafast -avoid_negative_ts 1 {segment_path}"
        subprocess.run(video_cmd, shell=True)

        # Вырезаем сегмент аудио
        print(f"Extracting audio segment...")
        audio_cmd = f"ffmpeg -ss {start:.3f} -to {end:.3f} -i {args.audio_path} -c:a pcm_s16le {audio_segment_path}"
        subprocess.run(audio_cmd, shell=True)

        # Обрабатываем сегмент с LatentSync
        process_cmd = f"python -m scripts.inference --unet_config_path 'configs/unet/stage2.yaml' --inference_ckpt_path 'checkpoints/latentsync_unet.pt' --inference_steps 30 --guidance_scale 1.5 --video_path '{segment_path}' --audio_path '{audio_segment_path}' --video_out_path '{output_path}'"
        try:
            print(f"Running inference on segment...")
            subprocess.run(process_cmd, shell=True, check=True)
            print(f"Segment processed successfully")

            # Обрезаем сегменты для склейки
            if i == 0:
                # Для первого сегмента берем только первые segment_length секунд
                trim_cmd = f"ffmpeg -i {output_path} -to {args.segment_length} -c:v copy -c:a copy {trimmed_path}"
                print(f"Trimming first segment to {args.segment_length} seconds")
            else:
                # Остальные сегменты копируем как есть
                shutil.copy(output_path, trimmed_path)
                print(f"Keeping segment as is")

            processed_files.append(trimmed_path)

        except subprocess.CalledProcessError as e:
            print(f"Error processing segment: {e}")
            # В случае ошибки пропускаем этот сегмент
            continue

    # Проверяем, есть ли обработанные файлы
    if not processed_files:
        print("Error: No segments were processed")
        shutil.rmtree(temp_dir)
        return

    # Создаем список файлов для объединения
    concat_file = os.path.join(temp_dir, "files.txt")
    with open(concat_file, "w") as f:
        for file_path in processed_files:
            # Используем относительные пути
            rel_path = os.path.relpath(file_path, temp_dir)
            f.write(f"file '{rel_path}'\n")

    # Объединяем результаты
    print("\nMerging processed segments...")
    merge_cmd = f"cd {temp_dir} && ffmpeg -f concat -safe 0 -i files.txt -c copy {os.path.abspath(args.output_path)}"
    subprocess.run(merge_cmd, shell=True)

    print(f"\nProcessing complete. Final output: {args.output_path}")

    # Сохраняем обработанные сегменты для анализа
    results_dir = "processed_segments"
    os.makedirs(results_dir, exist_ok=True)

    for i, file_path in enumerate(processed_files):
        dest_path = os.path.join(results_dir, f"segment_{i:03d}.mp4")
        shutil.copy(file_path, dest_path)

    print(f"Individual segments saved in '{results_dir}' directory")

    # Очистка временных файлов
    if not args.keep_temp:
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
    else:
        print(f"Temporary files preserved in directory: {temp_dir}")


if __name__ == "__main__":
    main()