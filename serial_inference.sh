#!/bin/bash

for project_dir in projects/*/; do
    project_name=$(basename "$project_dir")

    video_file=$(find "$project_dir" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.avi" \) -print -quit)

    audio_file="$project_dir/en_audio.mp3"

    if [ -z "$video_file" ]; then
        echo "❌ Error: $project_dir folder doesn't contain any video file"
        exit 1
    fi

    if [ ! -f "$audio_file" ]; then
        echo "❌ Error: $project_dir folder doesn't contain en_audio.mp3"
        exit 1
    fi

    echo "🚀 Starting processing folder: $project_name"
    python -m scripts.inference \
        --unet_config_path "configs/unet/stage2.yaml" \
        --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
        --inference_steps 50 \
        --guidance_scale 1.6 \
        --video_path "$video_file" \
        --audio_path "$audio_file" \
        --video_out_path "output/${project_name}_processed.mp4" || exit 1

    echo "✅ $project_name folder processed successfully!"
done

echo "🎉 All folders processed successfully!"
