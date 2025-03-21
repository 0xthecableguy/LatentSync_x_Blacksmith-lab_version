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

    # Получаем информацию о видео
    duration_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {args.video_path}"
    result = subprocess.run(duration_cmd, shell=True, capture_output=True, text=True)
    total_duration = float(result.stdout.strip())

    # Получаем точную частоту кадров
    fps_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 {args.video_path}"
    fps_result = subprocess.run(fps_cmd, shell=True, capture_output=True, text=True)
    fps_str = fps_result.stdout.strip()
    if '/' in fps_str:
        num, den = map(int, fps_str.split('/'))
        fps = num / den
    else:
        fps = float(fps_str)

    print(f"Total video duration: {total_duration:.2f} seconds, FPS: {fps}")

    # Рассчитываем точное количество кадров для сегментов
    segment_frames = int(args.segment_length * fps)
    overlap_frames = int(args.overlap * fps)

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

    # Создаем сегменты с перекрытием на основе точных временных меток
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

        # Рассчитываем точные кадры для этого сегмента
        start_frame = int(start * fps)
        end_frame = int(end * fps)

        segment_path = os.path.join(segments_dir, f"segment_{i:03d}.mp4")
        audio_segment_path = os.path.join(audio_segments_dir, f"audio_{i:03d}.wav")
        output_path = os.path.join(processed_dir, f"processed_{i:03d}.mp4")
        trimmed_path = os.path.join(trimmed_dir, f"trimmed_{i:03d}.mp4")

        # Вырезаем сегмент видео с точностью до кадра
        print(f"Extracting video segment from frame {start_frame} to {end_frame} (time: {start:.3f}s - {end:.3f}s)...")
        video_cmd = f"ffmpeg -ss {start:.6f} -to {end:.6f} -i {args.video_path} -c:v libx264 -preset ultrafast -avoid_negative_ts 1 -copyts {segment_path}"
        subprocess.run(video_cmd, shell=True)

        # Вырезаем сегмент аудио с высокой точностью
        print(f"Extracting audio segment...")
        audio_cmd = f"ffmpeg -ss {start:.6f} -to {end:.6f} -i {args.audio_path} -c:a pcm_s16le {audio_segment_path}"
        subprocess.run(audio_cmd, shell=True)

        # Обрабатываем сегмент с LatentSync
        process_cmd = f"python -m scripts.inference --unet_config_path 'configs/unet/stage2.yaml' --inference_ckpt_path 'checkpoints/latentsync_unet.pt' --inference_steps 50 --guidance_scale 2.0 --video_path '{segment_path}' --audio_path '{audio_segment_path}' --video_out_path '{output_path}'"
        try:
            print(f"Running inference on segment...")
            subprocess.run(process_cmd, shell=True, check=True)
            print(f"Segment processed successfully")

            # Обрезаем сегменты для склейки с точностью до кадра
            if i < len(segment_times) - 1:  # Если это не последний сегмент
                # Получаем точную длительность обработанного сегмента
                duration_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {output_path}"
                result = subprocess.run(duration_cmd, shell=True, capture_output=True, text=True)
                processed_duration = float(result.stdout.strip())

                # Получаем частоту кадров обработанного видео
                fps_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 {output_path}"
                fps_result = subprocess.run(fps_cmd, shell=True, capture_output=True, text=True)
                fps_str = fps_result.stdout.strip()
                if '/' in fps_str:
                    num, den = map(int, fps_str.split('/'))
                    processed_fps = num / den
                else:
                    processed_fps = float(fps_str)

                # Определяем длительность для обрезки с точностью до кадра
                # Обрезаем последние overlap секунд
                trim_duration = processed_duration - args.overlap

                # Убеждаемся, что длительность выравнена по кадрам
                frames_to_keep = int(trim_duration * processed_fps)
                trim_duration_aligned = frames_to_keep / processed_fps

                trim_cmd = f"ffmpeg -i {output_path} -t {trim_duration_aligned:.6f} -c:v copy -c:a copy {trimmed_path}"
                print(
                    f"Trimming segment {i} - keeping exactly {frames_to_keep} frames (duration: {trim_duration_aligned:.6f}s)")
                subprocess.run(trim_cmd, shell=True)
            else:
                # Последний сегмент оставляем как есть
                shutil.copy(output_path, trimmed_path)
                print(f"Keeping last segment as is")

            # Проверяем, что файл существует перед добавлением
            if os.path.exists(trimmed_path):
                processed_files.append(trimmed_path)
            else:
                print(f"Warning: Trimmed file {trimmed_path} does not exist")

        except subprocess.CalledProcessError as e:
            print(f"Error processing segment: {e}")
            # В случае ошибки пропускаем этот сегмент
            continue

    # Проверяем, есть ли обработанные файлы
    if not processed_files:
        print("Error: No segments were processed")
        if not args.keep_temp:
            shutil.rmtree(temp_dir)
        return

    # Создаем список файлов для объединения
    concat_file = os.path.join(temp_dir, "files.txt")
    with open(concat_file, "w") as f:
        for file_path in processed_files:
            # Используем абсолютные пути для надежности
            abs_path = os.path.abspath(file_path)
            f.write(f"file '{abs_path}'\n")

    # Объединяем результаты с точной синхронизацией
    print("\nMerging processed segments with precise alignment...")
    merge_cmd = f"ffmpeg -f concat -safe 0 -i {concat_file} -c copy -vsync 0 {args.output_path}"
    subprocess.run(merge_cmd, shell=True)

    print(f"\nProcessing complete. Final output: {args.output_path}")

    # Сохраняем обработанные сегменты для анализа
    results_dir = "processed_segments"
    os.makedirs(results_dir, exist_ok=True)

    for i, file_path in enumerate(processed_files):
        if os.path.exists(file_path):
            dest_path = os.path.join(results_dir, f"segment_{i:03d}.mp4")
            shutil.copy(file_path, dest_path)
            print(f"Saved segment {i} to {dest_path}")

    print(f"Individual segments saved in '{results_dir}' directory")

    # Очистка временных файлов
    if not args.keep_temp:
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
    else:
        print(f"Temporary files preserved in directory: {temp_dir}")


if __name__ == "__main__":
    main()