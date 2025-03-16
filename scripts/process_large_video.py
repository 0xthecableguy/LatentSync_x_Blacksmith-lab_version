import os
import subprocess
import argparse
import shutil
import json


def get_video_duration(video_path):
    """Получение длительности видео с помощью ffprobe"""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--segment_length", type=int, default=60, help="Length of each segment in seconds")
    args = parser.parse_args()

    # Определяем текущую директорию скрипта и корневую директорию проекта
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Предполагаем, что скрипт находится в подкаталоге проекта

    # Настраиваем пути относительно корня проекта
    inference_script = os.path.join(project_root, "scripts", "inference.py")
    config_path = os.path.join(project_root, "configs/unet/second_stage.yaml")
    checkpoint_path = os.path.join(project_root, "checkpoints/latentsync_unet.pt")

    # Проверка наличия файлов
    if not os.path.exists(inference_script):
        print(f"Error: Inference script not found at {inference_script}")
        print(f"Current directory: {os.getcwd()}")
        print("Available files in scripts directory:")
        scripts_dir = os.path.join(project_root, "scripts")
        if os.path.exists(scripts_dir):
            print(os.listdir(scripts_dir))
        return

    # Очистка и создание временной директории
    temp_dir = os.path.join(project_root, "temp_segments")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Получаем длительность исходного видео
    video_duration = get_video_duration(args.video_path)
    print(f"Video duration: {video_duration:.2f} seconds")

    # Создаем список позиций сегментов
    segment_positions = []
    current_position = 0

    while current_position < video_duration:
        end_position = min(current_position + args.segment_length, video_duration)
        segment_positions.append((current_position, end_position))
        current_position = end_position

    print(f"Will process video in {len(segment_positions)} segments")

    # Файл для списка обработанных сегментов
    segments_list_file = os.path.join(temp_dir, "processed_segments.txt")
    with open(segments_list_file, "w") as f:
        f.write("")  # Создаем пустой файл

    # Обрабатываем каждый сегмент последовательно
    for i, (start, end) in enumerate(segment_positions):
        print(f"\nProcessing segment {i + 1}/{len(segment_positions)}: {start:.2f}s - {end:.2f}s")

        # Очищаем предыдущие временные файлы для экономии места
        for temp_file in os.listdir(temp_dir):
            if temp_file != "processed_segments.txt" and not temp_file.startswith("output_"):
                file_path = os.path.join(temp_dir, temp_file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Создаем имена файлов для этого сегмента
        segment_path = os.path.join(temp_dir, f"segment.mp4")
        audio_segment_path = os.path.join(temp_dir, f"audio.wav")
        output_path = os.path.join(temp_dir, f"output_{i:03d}.mp4")

        # Вырезаем сегмент видео
        print(f"Extracting video segment from {start:.2f}s to {end:.2f}s...")
        video_cmd = f"ffmpeg -y -ss {start:.3f} -to {end:.3f} -i {args.video_path} -c:v libx264 -preset ultrafast -crf 18 {segment_path}"
        subprocess.run(video_cmd, shell=True)

        # Вырезаем соответствующий сегмент аудио
        print(f"Extracting audio segment...")
        audio_cmd = f"ffmpeg -y -ss {start:.3f} -to {end:.3f} -i {args.audio_path} -c:a pcm_s16le {audio_segment_path}"
        subprocess.run(audio_cmd, shell=True)

        # Запускаем обработку данного сегмента
        print(f"Processing segment with LatentSync...")
        # Используем полный путь к inference.py и запускаем его как отдельный скрипт
        process_cmd = (
            f"python {inference_script} "
            f"--unet_config_path '{config_path}' "
            f"--inference_ckpt_path '{checkpoint_path}' "
            f"--inference_steps 30 --guidance_scale 1.5 "
            f"--video_path '{segment_path}' "
            f"--audio_path '{audio_segment_path}' "
            f"--video_out_path '{output_path}'"
        )
        subprocess.run(process_cmd, shell=True, check=False)  # Не останавливаемся при ошибке

        # Проверяем, был ли создан выходной файл
        if not os.path.exists(output_path):
            print(f"Warning: Output file {output_path} was not created. Skipping this segment.")
            continue

        # Удаляем временные файлы сегмента после обработки
        os.remove(segment_path)
        os.remove(audio_segment_path)

        # Добавляем обработанный сегмент в список
        with open(segments_list_file, "a") as f:
            f.write(f"file '{os.path.abspath(output_path)}'\n")

    # Проверяем, есть ли сегменты для объединения
    with open(segments_list_file, "r") as f:
        if not f.read().strip():
            print("Error: No segments were processed successfully.")
            shutil.rmtree(temp_dir)
            return

    # Объединяем результаты
    print("\nMerging all processed segments...")
    merge_cmd = f"ffmpeg -f concat -safe 0 -i {segments_list_file} -c copy {args.output_path}"
    subprocess.run(merge_cmd, shell=True)

    print(f"\nProcessing complete! Final output: {args.output_path}")

    # Удаляем временные файлы
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()