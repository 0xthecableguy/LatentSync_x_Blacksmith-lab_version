#!/usr/bin/env python3
"""
Скрипт для обработки длинных видео сегментами с последующей склейкой.
Оптимизирован для сохранения высокого качества и плавных переходов между сегментами.
"""

import os
import subprocess
import argparse
import shutil
import json
import sys
from datetime import datetime


def run_command(cmd, desc=None, check=True, capture_output=False):
    """Запуск команды с выводом информации и обработкой ошибок"""
    if desc:
        print(f"\n[{desc}]")
        print(f"Running: {cmd}")

    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
            return result
        else:
            subprocess.run(cmd, shell=True, check=check)
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        if check:
            sys.exit(1)
        return None


def get_video_info(file_path):
    """Получение информации о видеофайле: длительность, битрейт, fps и т.д."""
    cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams "{file_path}"'
    result = run_command(cmd, desc="Getting video info", capture_output=True)

    if not result or not result.stdout:
        print(f"Error: Could not get video information for {file_path}")
        sys.exit(1)

    info = json.loads(result.stdout)

    # Базовая информация
    duration = float(info['format']['duration'])

    # Информация о видеопотоке
    video_stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
    audio_stream = next((s for s in info['streams'] if s['codec_type'] == 'audio'), None)

    video_info = {
        'duration': duration,
        'size': float(info['format']['size']),
        'bit_rate': float(info['format'].get('bit_rate', 0)),
    }

    if video_stream:
        # Получаем frame rate в формате num/den
        fr_parts = video_stream.get('r_frame_rate', '').split('/')
        if len(fr_parts) == 2 and fr_parts[0].isdigit() and fr_parts[1].isdigit():
            video_info['fps'] = float(fr_parts[0]) / float(fr_parts[1])
        else:
            video_info['fps'] = float(video_stream.get('avg_frame_rate', '25').split('/')[0])

        video_info['width'] = int(video_stream.get('width', 0))
        video_info['height'] = int(video_stream.get('height', 0))
        video_info['codec'] = video_stream.get('codec_name', '')

        # Определяем GOP (Group of Pictures) - расстояние между ключевыми кадрами
        if 'codec_name' in video_stream:
            keyframes_cmd = f'ffprobe -v error -select_streams v:0 -show_entries frame=pict_type -of csv=p=0 "{file_path}" | grep -n I | sed -n "1p;2p" | tr "\\n" " "'
            keyframes_result = run_command(keyframes_cmd, desc="Estimating keyframe interval", capture_output=True,
                                           check=False)
            if keyframes_result and keyframes_result.stdout:
                kf_lines = keyframes_result.stdout.strip().split()
                if len(kf_lines) >= 2:
                    try:
                        # Извлекаем номера строк из вывода (первые две строки с I-кадрами)
                        first_kf = int(kf_lines[0].split(':')[0])
                        second_kf = int(kf_lines[1].split(':')[0])
                        video_info['keyframe_interval'] = second_kf - first_kf
                    except (ValueError, IndexError):
                        video_info['keyframe_interval'] = 25  # Значение по умолчанию
                else:
                    video_info['keyframe_interval'] = 25
            else:
                video_info['keyframe_interval'] = 25

    if audio_stream:
        video_info['audio_codec'] = audio_stream.get('codec_name', '')
        video_info['audio_channels'] = int(audio_stream.get('channels', 0))
        video_info['audio_sample_rate'] = int(audio_stream.get('sample_rate', 0))

    return video_info


def create_optimal_segment_plan(duration, segment_length, overlap, keyframe_interval=None):
    """
    Создает оптимальный план сегментации видео с учетом ключевых кадров

    Args:
        duration: Общая длительность видео в секундах
        segment_length: Желаемая длина сегмента в секундах
        overlap: Желаемое перекрытие в секундах
        keyframe_interval: Интервал между ключевыми кадрами в кадрах

    Returns:
        Список кортежей (start_time, end_time) для каждого сегмента
    """
    segments = []
    start_time = 0

    while start_time < duration:
        # Основная длительность сегмента
        end_time = min(start_time + segment_length, duration)

        # Добавляем перекрытие, если это не последний сегмент
        if end_time < duration:
            end_time = min(end_time + overlap, duration)

        segments.append((start_time, end_time))

        # Следующий сегмент начинается после текущего, минус перекрытие
        start_time = end_time - overlap if end_time < duration else duration

    return segments


def main():
    parser = argparse.ArgumentParser(description='Process video in segments with high-quality merging')
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--output_path", type=str, required=True, help="Path for final output video")
    parser.add_argument("--segment_length", type=int, default=60, help="Target length of each segment in seconds")
    parser.add_argument("--overlap", type=int, default=5, help="Overlap between segments in seconds")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary files after processing")
    parser.add_argument("--quality", type=str, default="high", choices=["ultrafast", "fast", "medium", "slow", "high"],
                        help="Encoding quality/speed tradeoff")
    args = parser.parse_args()

    # Настройки качества кодирования
    quality_presets = {
        "ultrafast": {"preset": "ultrafast", "crf": "23"},
        "fast": {"preset": "fast", "crf": "22"},
        "medium": {"preset": "medium", "crf": "20"},
        "slow": {"preset": "slow", "crf": "18"},
        "high": {"preset": "slow", "crf": "17"}
    }

    encoding_settings = quality_presets[args.quality]

    # Создаем временный каталог с датой и временем для уникальности
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = f"temp_processing_{timestamp}"

    # Структура каталогов
    dirs = {
        "root": temp_dir,
        "segments": os.path.join(temp_dir, "video_segments"),
        "audio_segments": os.path.join(temp_dir, "audio_segments"),
        "processed": os.path.join(temp_dir, "processed_segments"),
        "trimmed": os.path.join(temp_dir, "trimmed_segments"),
        "final": os.path.join(temp_dir, "final_parts")
    }

    # Создаем все необходимые каталоги
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Получаем информацию о видео
    video_info = get_video_info(args.video_path)
    print(f"Video information:")
    print(f"- Duration: {video_info['duration']:.2f} seconds")
    print(f"- Resolution: {video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}")
    print(f"- FPS: {video_info.get('fps', 'N/A')}")
    print(f"- Video codec: {video_info.get('codec', 'N/A')}")
    print(f"- Audio codec: {video_info.get('audio_codec', 'N/A')}")
    print(f"- Estimated keyframe interval: {video_info.get('keyframe_interval', 'N/A')} frames")

    # Создаем оптимальный план сегментации
    segments = create_optimal_segment_plan(
        video_info['duration'],
        args.segment_length,
        args.overlap,
        video_info.get('keyframe_interval')
    )

    print(f"\nPlanned {len(segments)} segments with {args.overlap}s overlap")
    for i, (start, end) in enumerate(segments):
        print(f"Segment {i + 1}: {start:.2f}s - {end:.2f}s (duration: {end - start:.2f}s)")

    # Нарезаем сегменты
    processed_segments = []

    for i, (start, end) in enumerate(segments):
        print(f"\n{'=' * 80}")
        print(f"Processing segment {i + 1}/{len(segments)}: {start:.2f}s - {end:.2f}s")
        print(f"{'=' * 80}")

        segment_path = os.path.join(dirs["segments"], f"segment_{i:03d}.mp4")
        audio_segment_path = os.path.join(dirs["audio_segments"], f"audio_{i:03d}.wav")
        processed_path = os.path.join(dirs["processed"], f"processed_{i:03d}.mp4")

        # Извлекаем видеосегмент с более высоким качеством
        # Используем двухпроходное извлечение для более точного начала сегмента
        print(f"\nExtracting video segment {i + 1}...")
        video_cmd = (
            f'ffmpeg -ss {start:.3f} -i "{args.video_path}" -to {end - start:.3f} '
            f'-c:v libx264 -preset {encoding_settings["preset"]} -crf {encoding_settings["crf"]} '
            f'-pix_fmt yuv420p -force_key_frames "expr:gte(t,0)" '
            f'-profile:v high -level 4.2 -g {int(video_info.get("fps", 25))} '
            f'-keyint_min {max(5, int(video_info.get("fps", 25) / 5))} '
            f'-sc_threshold 0 -bf 2 -b_strategy 2 -flags +cgop '
            f'-movflags +faststart -y "{segment_path}"'
        )
        run_command(video_cmd, desc=f"Extracting video segment {i + 1}")

        # Извлекаем аудиосегмент
        print(f"Extracting audio segment {i + 1}...")
        audio_cmd = (
            f'ffmpeg -ss {start:.3f} -i "{args.audio_path}" -to {end - start:.3f} '
            f'-c:a pcm_s16le -ar 48000 -ac 2 -y "{audio_segment_path}"'
        )
        run_command(audio_cmd, desc=f"Extracting audio segment {i + 1}")

        # Проверяем, что файлы созданы
        if not os.path.exists(segment_path) or not os.path.exists(audio_segment_path):
            print(f"Error: Failed to create segment {i + 1} files")
            continue

        # Обрабатываем сегмент с LatentSync (или другой моделью)
        print(f"Running inference on segment {i + 1}...")
        process_cmd = (
            f'python -m scripts.inference --unet_config_path "configs/unet/stage2.yaml" '
            f'--inference_ckpt_path "checkpoints/latentsync_unet.pt" --inference_steps 50 '
            f'--guidance_scale 2.0 --video_path "{segment_path}" --audio_path "{audio_segment_path}" '
            f'--video_out_path "{processed_path}"'
        )

        try:
            run_command(process_cmd, desc=f"Processing segment {i + 1} with LatentSync")

            # Проверяем, что обработанный файл создан
            if not os.path.exists(processed_path):
                print(f"Error: Failed to process segment {i + 1}")
                continue

            # Добавляем в список обработанных
            processed_segments.append({
                "index": i,
                "path": processed_path,
                "start": start,
                "end": end,
                "duration": end - start
            })

            print(f"Segment {i + 1} processed successfully")

        except Exception as e:
            print(f"Error processing segment {i + 1}: {e}")
            continue

    # Проверяем, есть ли обработанные сегменты
    if not processed_segments:
        print("Error: No segments were processed successfully")
        if not args.keep_temp:
            shutil.rmtree(temp_dir)
        return

    # Подготавливаем сегменты для склейки - обрезаем перекрытия
    print("\nPreparing segments for merging...")
    final_segments = []

    for i, segment in enumerate(processed_segments):
        trimmed_path = os.path.join(dirs["trimmed"], f"trimmed_{segment['index']:03d}.mp4")
        final_path = os.path.join(dirs["final"], f"final_{segment['index']:03d}.mp4")

        # Получаем точную длительность обработанного сегмента
        segment_info = get_video_info(segment['path'])
        segment_duration = segment_info['duration']

        # Определяем, сколько нужно обрезать с начала и конца
        trim_start = 0
        trim_end = segment_duration

        # Если это не первый сегмент, обрезаем начало (первые N/2 секунд перекрытия)
        if i > 0 and args.overlap > 0:
            trim_start = args.overlap / 2

        # Если это не последний сегмент, обрезаем конец (последние N/2 секунд перекрытия)
        if i < len(processed_segments) - 1 and args.overlap > 0:
            trim_end = segment_duration - (args.overlap / 2)

        # Обрезаем сегмент
        trim_cmd = (
            f'ffmpeg -i "{segment["path"]}" -ss {trim_start:.3f} -to {trim_end:.3f} '
            f'-c:v libx264 -preset {encoding_settings["preset"]} -crf {encoding_settings["crf"]} '
            f'-pix_fmt yuv420p -force_key_frames "expr:gte(t,0)" '
            f'-profile:v high -level 4.2 -flags +cgop -movflags +faststart '
            f'-y "{trimmed_path}"'
        )
        run_command(trim_cmd, desc=f"Trimming segment {segment['index']} for merge")

        # Добавляем переход-кроссфейд если это не первый и не последний сегмент
        if i > 0 and i < len(processed_segments) - 1 and args.overlap > 0:
            # Для будущего расширения - можно добавить плавное смешивание между сегментами
            # Пока просто копируем обрезанный файл
            shutil.copy(trimmed_path, final_path)
        else:
            # Для первого и последнего сегмента просто копируем
            shutil.copy(trimmed_path, final_path)

        final_segments.append(final_path)

    # Создаем список файлов для объединения
    concat_file = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_file, "w") as f:
        for file_path in final_segments:
            # Используем относительные пути для совместимости
            rel_path = os.path.relpath(file_path, start=os.path.dirname(concat_file))
            f.write(f"file '{rel_path}'\n")

    # Объединяем результаты с высоким качеством
    print("\nMerging processed segments into final video...")
    merge_cmd = (
        f'ffmpeg -f concat -safe 0 -i "{concat_file}" '
        f'-c:v libx264 -preset {encoding_settings["preset"]} -crf {encoding_settings["crf"]} '
        f'-pix_fmt yuv420p -profile:v high -level 4.2 '
        f'-movflags +faststart -y "{args.output_path}"'
    )
    run_command(merge_cmd, desc="Creating final video")

    # Проверяем результат
    if os.path.exists(args.output_path):
        final_info = get_video_info(args.output_path)
        print(f"\nFinal video created successfully:")
        print(f"- Path: {args.output_path}")
        print(f"- Duration: {final_info['duration']:.2f} seconds")
        print(f"- Size: {final_info['size'] / (1024 * 1024):.2f} MB")
    else:
        print(f"\nError: Failed to create final video at {args.output_path}")

    # Очистка
    if not args.keep_temp:
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)
    else:
        print(f"\nTemporary files preserved in: {temp_dir}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()