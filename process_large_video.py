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

    # Getting information about the video duration
    duration_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {args.video_path}"
    result = subprocess.run(duration_cmd, shell=True, capture_output=True, text=True)
    total_duration = float(result.stdout.strip())
    print(f"Total video duration: {total_duration:.2f} seconds")

    # Cleaning and creating a temporary directory
    temp_dir = "temp_segments"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Create segments with precise time indication
    segment_times = []
    start_time = 0
    while start_time < total_duration:
        end_time = min(start_time + args.segment_length, total_duration)
        segment_times.append((start_time, end_time))
        start_time = end_time

    print(f"Will create {len(segment_times)} segments")

    output_files = []

    # Processing each segment
    for i, (start, end) in enumerate(segment_times):
        print(f"\nProcessing segment {i + 1}/{len(segment_times)}: {start:.2f}s - {end:.2f}s")

        segment_path = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
        audio_segment_path = os.path.join(temp_dir, f"audio_{i:03d}.mp3")
        output_path = os.path.join(temp_dir, f"output_{i:03d}.mp4")
        output_files.append(output_path)

        # Cutting out a segment of the video with precise timestamps
        print(f"Extracting video segment from {start:.2f}s to {end:.2f}s...")
        video_cmd = f"ffmpeg -ss {start:.3f} -to {end:.3f} -i {args.video_path} -c:v libx264 -preset ultrafast -avoid_negative_ts 1 {segment_path}"
        subprocess.run(video_cmd, shell=True)

        # Cutting out the corresponding audio segment
        print(f"Extracting audio segment...")
        audio_cmd = f"ffmpeg -ss {start:.3f} -to {end:.3f} -i {args.audio_path} -c:a aac {audio_segment_path}"
        subprocess.run(audio_cmd, shell=True)

        # Processing the segment using LatentSync
        process_cmd = f"python -m scripts.inference --unet_config_path 'configs/unet/stage2.yaml' --inference_ckpt_path 'checkpoints/latentsync_unet.pt' --inference_steps 30 --guidance_scale 1.5 --video_path '{segment_path}' --audio_path '{audio_segment_path}' --video_out_path '{output_path}'"
        try:
            print(f"Running inference on segment...")
            subprocess.run(process_cmd, shell=True, check=True)
            print(f"Segment processed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error processing segment: {e}")
            continue

        # Checking if the output file has been created
        if not os.path.exists(output_path):
            print(f"Warning: Output file {output_path} was not created")
            continue

    # Checking if there are any processed files
    processed_files = [f for f in output_files if os.path.exists(f)]
    if not processed_files:
        print("Error: No segments were processed")
        shutil.rmtree(temp_dir)
        return

    # Using a reliable method to combine videos
    # Creating a list of files for concat demuxer
    with open(f"{temp_dir}/files.txt", "w") as f:
        for output_file in processed_files:
            # Using relative paths relative to files.txt
            rel_path = os.path.relpath(output_file, temp_dir)
            # Escaping apostrophes on the way
            safe_path = rel_path.replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    # Merging the results
    print("\nMerging processed segments...")
    merge_cmd = f"cd {temp_dir} && ffmpeg -f concat -safe 0 -i files.txt -c copy {os.path.abspath(args.output_path)}"
    subprocess.run(merge_cmd, shell=True)

    print(f"\nProcessing complete. Final output: {args.output_path}")

    # Cleaning temporary files
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()