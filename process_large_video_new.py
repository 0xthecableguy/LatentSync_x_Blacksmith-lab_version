import os
import subprocess
import argparse
import shutil
import json


def get_video_duration(video_path):
    """Получает точную длительность видео в секундах."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration", "-of", "json", video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    data = json.loads(result.stdout)

    # Если stream duration недоступна, попробуем duration формата
    if "streams" in data and data["streams"] and "duration" in data["streams"][0]:
        return float(data["streams"][0]["duration"])
    else:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])


def get_video_info(video_path):
    """Получает подробную информацию о видео."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "json", video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--segment_length", type=int, default=60, help="Length of each segment in seconds")
    parser.add_argument("--overlap", type=int, default=5, help="Overlap between segments in seconds")
    args = parser.parse_args()

    # Получаем информацию о длительности видео
    total_duration = get_video_duration(args.video_path)
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
    trimmed_files = []

    for i, (start, end) in enumerate(segment_times):
        print(f"\nProcessing segment {i + 1}/{len(segment_times)}: {start:.2f}s - {end:.2f}s")

        segment_path = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
        audio_segment_path = os.path.join(temp_dir, f"audio_{i:03d}.mp3")
        output_path = os.path.join(temp_dir, f"output_{i:03d}.mp4")
        trimmed_path = os.path.join(temp_dir, f"trimmed_{i:03d}.mp4")
        processed_files.append(output_path)
        trimmed_files.append(trimmed_path)

        # Вырезаем сегмент видео с точными временными метками
        print(f"Extracting video segment from {start:.2f}s to {end:.2f}s...")
        video_cmd = f"ffmpeg -ss {start:.3f} -to {end:.3f} -i {args.video_path} -c:v libx264 -preset ultrafast -force_key_frames 0 -avoid_negative_ts 1 {segment_path}"
        subprocess.run(video_cmd, shell=True)

        # Проверяем длительность вырезанного сегмента
        segment_duration = get_video_duration(segment_path)
        print(f"Extracted segment duration: {segment_duration:.2f} seconds")

        # Вырезаем соответствующий сегмент аудио
        print(f"Extracting audio segment...")
        audio_cmd = f"ffmpeg -ss {start:.3f} -to {end:.3f} -i {args.audio_path} -c:a aac {audio_segment_path}"
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

        # Получаем длительность обработанного файла
        processed_duration = get_video_duration(output_path)
        print(f"Processed segment duration: {processed_duration:.2f} seconds")

        # Вычисляем время обрезки для перекрытия с учетом пропорций
        # Обрабатываем случай, когда длительность обработанного сегмента
        # отличается от длительности исходного сегмента
        expected_duration = end - start
        ratio = processed_duration / expected_duration if expected_duration > 0 else 1.0

        trim_start = 0
        trim_duration = processed_duration

        if i > 0:  # Не первый сегмент - обрезаем начало
            trim_start = (args.overlap / 2) * ratio

        if i < len(segment_times) - 1:  # Не последний сегмент - обрезаем конец
            trim_duration = processed_duration - (args.overlap / 2) * ratio

        print(f"Trimming segment: start={trim_start:.2f}s, duration={trim_duration:.2f}s")

        # Обрезаем файл с точными временными метками
        trim_cmd = f"ffmpeg -i {output_path} -ss {trim_start:.3f} -t {trim_duration - trim_start:.3f} -c:v libx264 -preset fast -c:a aac {trimmed_path}"
        subprocess.run(trim_cmd, shell=True)

        # Проверяем длительность обрезанного файла
        trimmed_duration = get_video_duration(trimmed_path)
        print(f"Trimmed segment duration: {trimmed_duration:.2f} seconds")

        if not os.path.exists(trimmed_path) or trimmed_duration <= 0:
            print(f"Warning: Trimmed file {trimmed_path} was not created correctly, using original")
            shutil.copy(output_path, trimmed_path)

    # Убеждаемся, что у нас есть обработанные файлы
    existing_trimmed_files = [f for f in trimmed_files if os.path.exists(f)]
    if not existing_trimmed_files:
        print("Error: No segments were processed")
        shutil.rmtree(temp_dir)
        return

    # Выводим информацию о всех trimmed файлах
    print("\nAll trimmed files information:")
    for file_path in existing_trimmed_files:
        duration = get_video_duration(file_path)
        print(f"{os.path.basename(file_path)}: {duration:.2f} seconds")

    # Создаем список файлов для объединения
    concat_file = os.path.join(temp_dir, "files.txt")
    with open(concat_file, "w") as f:
        for file_path in existing_trimmed_files:
            # Используем относительные пути
            rel_path = os.path.basename(file_path)
            f.write(f"file '{rel_path}'\n")

    # Объединяем видео сегменты без аудио
    print("\nMerging processed video segments without audio...")
    temp_video_without_audio = os.path.join(temp_dir, "temp_video_no_audio.mp4")
    merge_cmd = f"cd {temp_dir} && ffmpeg -f concat -safe 0 -i files.txt -c copy {os.path.basename(temp_video_without_audio)}"
    subprocess.run(merge_cmd, shell=True)

    # Если видео без аудио не создалось, завершаем с ошибкой
    if not os.path.exists(temp_video_without_audio):
        print("Error: Failed to create merged video without audio")
        shutil.rmtree(temp_dir)
        return

    # Проверяем длительность объединенного видео
    combined_duration = get_video_duration(temp_video_without_audio)
    print(f"Combined video duration: {combined_duration:.2f} seconds")

    # Добавляем полную оригинальную аудиодорожку к финальному видео
    print("\nAdding original audio to the final video...")
    final_cmd = f"ffmpeg -i {temp_video_without_audio} -i {args.audio_path} -map 0:v -map 1:a -c:v copy -c:a aac -shortest {args.output_path}"
    subprocess.run(final_cmd, shell=True)

    print(f"\nProcessing complete. Final output: {args.output_path}")

    # Проверяем длительность финального видео
    final_duration = get_video_duration(args.output_path)
    print(f"Final video duration: {final_duration:.2f} seconds")

    # Очистка временных файлов
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()