import os
import subprocess
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--segment_length", type=int, default=60, help="Length of each segment in seconds")
    parser.add_argument("--overlap", type=int, default=10, help="Overlap for segments in seconds")
    args = parser.parse_args()

    # Получаем информацию о длительности видео
    duration_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {args.video_path}"
    result = subprocess.run(duration_cmd, shell=True, capture_output=True, text=True)
    total_duration = float(result.stdout.strip())
    print(f"Total video duration: {total_duration:.2f} seconds")

    # Создаем директории для сегментов
    segments_dir = "video_segments"
    audio_segments_dir = "audio_segments"

    if os.path.exists(segments_dir):
        shutil.rmtree(segments_dir)
    if os.path.exists(audio_segments_dir):
        shutil.rmtree(audio_segments_dir)

    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(audio_segments_dir, exist_ok=True)

    # Создаем сегменты с перекрытием
    segment_times = []
    start_time = 0
    overlap_start = args.segment_length - args.overlap  # Начало перекрытия (для сегмента 60 сек и перекрытия 10 сек это будет 50-я секунда)

    while start_time < total_duration:
        end_time = min(start_time + args.segment_length + args.overlap, total_duration)
        segment_times.append((start_time, end_time))
        start_time += args.segment_length - args.overlap  # Следующий сегмент начинается с учетом перекрытия

    print(f"Will create {len(segment_times)} segments with {args.overlap}s overlap at each end")

    # Нарезаем сегменты
    for i, (start, end) in enumerate(segment_times):
        print(f"\nCreating segment {i + 1}/{len(segment_times)}: {start:.2f}s - {end:.2f}s")

        segment_path = os.path.join(segments_dir, f"segment_{i:03d}.mp4")
        audio_segment_path = os.path.join(audio_segments_dir, f"audio_{i:03d}.wav")

        # Вырезаем сегмент видео
        video_cmd = f"ffmpeg -ss {start:.3f} -to {end:.3f} -i {args.video_path} -c:v libx264 -preset ultrafast -avoid_negative_ts 1 {segment_path}"
        subprocess.run(video_cmd, shell=True)

        # Вырезаем сегмент аудио
        audio_cmd = f"ffmpeg -ss {start:.3f} -to {end:.3f} -i {args.audio_path} -c:a pcm_s16le {audio_segment_path}"
        subprocess.run(audio_cmd, shell=True)

        # Проверяем длительность полученных файлов
        video_duration_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {segment_path}"
        video_result = subprocess.run(video_duration_cmd, shell=True, capture_output=True, text=True)
        video_duration = float(video_result.stdout.strip())

        audio_duration_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {audio_segment_path}"
        audio_result = subprocess.run(audio_duration_cmd, shell=True, capture_output=True, text=True)
        audio_duration = float(audio_result.stdout.strip())

        print(f"  Segment {i}: Video duration = {video_duration:.3f}s, Audio duration = {audio_duration:.3f}s")

        # Создаем файл с информацией о сегменте
        with open(os.path.join(segments_dir, f"segment_{i:03d}_info.txt"), "w") as f:
            f.write(f"Original start time: {start:.3f}s\n")
            f.write(f"Original end time: {end:.3f}s\n")
            f.write(f"Resulting video duration: {video_duration:.3f}s\n")
            f.write(f"Resulting audio duration: {audio_duration:.3f}s\n")

            if i > 0:
                f.write(f"Overlap with previous segment starts at: {start:.3f}s (original timeline)\n")
                f.write(f"Overlap with previous segment ends at: {start + args.overlap:.3f}s (original timeline)\n")

            if i < len(segment_times) - 1:
                f.write(f"Overlap with next segment starts at: {end - args.overlap:.3f}s (original timeline)\n")
                f.write(f"Overlap with next segment ends at: {end:.3f}s (original timeline)\n")

    print(f"\nAll segments created successfully!")
    print(f"Video segments saved in '{segments_dir}' directory")
    print(f"Audio segments saved in '{audio_segments_dir}' directory")
    print(f"Each segment has info file with timing details for verification")


if __name__ == "__main__":
    main()