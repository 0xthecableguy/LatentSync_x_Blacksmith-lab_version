import os
import subprocess
import argparse
import shutil
import glob
import json


def get_video_duration(video_path):
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
    parser.add_argument("--overlap", type=int, default=2, help="Overlap between segments in seconds")
    args = parser.parse_args()

    temp_dir = "temp_segments"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    video_duration = get_video_duration(args.video_path)
    print(f"Video duration: {video_duration:.2f} seconds")

    segment_positions = []
    current_position = 0

    while current_position < video_duration:
        end_position = min(current_position + args.segment_length, video_duration)
        segment_positions.append((current_position, end_position))
        current_position = end_position - args.overlap

    print(f"Created {len(segment_positions)} segments for processing")

    output_files = []

    for i, (start, end) in enumerate(segment_positions):
        segment_path = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
        audio_segment_path = os.path.join(temp_dir, f"audio_{i:03d}.wav")
        output_path = os.path.join(temp_dir, f"output_{i:03d}.mp4")
        output_files.append((output_path, start, end))

        video_cmd = (
            f"ffmpeg -ss {start:.3f} -to {end:.3f} -i {args.video_path} "
            f"-c:v libx264 -preset ultrafast -crf 18 -c:a aac {segment_path}"
        )
        subprocess.run(video_cmd, shell=True, check=True)

        audio_cmd = (
            f"ffmpeg -ss {start:.3f} -to {end:.3f} -i {args.audio_path} "
            f"-c:a pcm_s16le {audio_segment_path}"
        )
        subprocess.run(audio_cmd, shell=True, check=True)

        process_cmd = (
            f"python -m scripts.inference "
            f"--unet_config_path 'configs/unet/second_stage.yaml' "
            f"--inference_ckpt_path 'checkpoints/latentsync_unet.pt' "
            f"--inference_steps 50 --guidance_scale 1.5 "
            f"--video_path '{segment_path}' "
            f"--audio_path '{audio_segment_path}' "
            f"--video_out_path '{output_path}'"
        )

        print(f"\nProcessing segment {i + 1}/{len(segment_positions)}: {start:.2f}s - {end:.2f}s")
        subprocess.run(process_cmd, shell=True, check=True)

    final_assembly_dir = os.path.join(temp_dir, "final_assembly")
    os.makedirs(final_assembly_dir, exist_ok=True)

    filter_complex = []
    inputs = []

    for i, (output_file, start, end) in enumerate(output_files):
        if i == 0:
            target_end = end - args.overlap / 2 if i < len(output_files) - 1 else end
            filter_complex.append(f"[0:v]trim=end={target_end - start}[v{i}]")
        elif i == len(output_files) - 1:
            filter_complex.append(f"[{i}:v]trim=start={args.overlap / 2}[v{i}]")
        else:
            segment_duration = end - start
            trim_start = args.overlap / 2
            trim_end = segment_duration - args.overlap / 2
            filter_complex.append(f"[{i}:v]trim=start={trim_start}:end={trim_end}[v{i}]")

        inputs.append(f"-i {output_file}")

    if len(output_files) > 1:
        for i in range(len(output_files) - 1):
            filter_complex.append(f"[v{i}][v{i + 1}]concat=n=2:v=1:a=0[v{i + 1}_concat]")

        final_output = f"[v{len(output_files) - 1}_concat]"
    else:
        final_output = "[v0]"

    filter_text = ';'.join(filter_complex)
    inputs_text = ' '.join(inputs)

    with open(f"{temp_dir}/files.txt", "w") as f:
        for output_file, _, _ in output_files:
            f.write(f"file '{os.path.abspath(output_file)}'\n")

    merge_cmd = f"ffmpeg -f concat -safe 0 -i {temp_dir}/files.txt -c copy {args.output_path}"
    subprocess.run(merge_cmd, shell=True, check=True)

    print(f"\nProcessing complete. Final output: {args.output_path}")

    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()