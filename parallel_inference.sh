#!/bin/bash

NUM_FOLDERS=11

MAX_PROCESSES=3

LOG_FILE="processing_log.txt"
echo "Processing started at $(date)" > $LOG_FILE

declare -A pid_folder

for i in $(seq 1 $NUM_FOLDERS); do
  if [ -d "assets/$i" ]; then
    if [ -f "assets/$i/$i.mp4" ] && [ -f "assets/$i/output_stereo.mp3" ]; then
      echo "Queueing folder assets/$i"

      mkdir -p "output/$i"

      while [ $(jobs -p | wc -l) -ge $MAX_PROCESSES ]; do
        echo "Waiting for a process to finish..."
        sleep 5
      done

      echo "Starting processing of assets/$i at $(date)" >> $LOG_FILE
      python -m scripts.inference \
        --unet_config_path "configs/unet/stage2.yaml" \
        --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
        --inference_steps 50 \
        --guidance_scale 1.6 \
        --video_path "assets/$i/$i.mp4" \
        --audio_path "assets/$i/output_stereo.mp3" \
        --video_out_path "output/$i/video_out.mp4" > "output/$i/process.log" 2>&1 &

      pid=$!
      pid_folder[$pid]=$i
      echo "Started process $pid for folder assets/$i" >> $LOG_FILE

      sleep 2
    else
      echo "Skipping folder assets/$i: Required files not found" | tee -a $LOG_FILE
    fi
  else
    echo "Folder assets/$i does not exist" | tee -a $LOG_FILE
  fi
done

pid_exists() {
  ps -p $1 > /dev/null
  return $?
}

echo "Waiting for all processes to complete..." | tee -a $LOG_FILE
for pid in "${!pid_folder[@]}"; do
  folder=${pid_folder[$pid]}
  wait $pid
  exit_status=$?

  if [ $exit_status -eq 0 ]; then
    echo "Process $pid for folder assets/$folder completed successfully at $(date)" | tee -a $LOG_FILE
    if [ -f "output/$folder/video_out.mp4" ] && [ -s "output/$folder/video_out.mp4" ]; then
      echo "Output file for folder $folder created successfully" | tee -a $LOG_FILE
    else
      echo "WARNING: Output file for folder $folder may be missing or empty!" | tee -a $LOG_FILE
    fi
  else
    echo "ERROR: Process $pid for folder assets/$folder failed with exit code $exit_status at $(date)" | tee -a $LOG_FILE
    echo "Check output/$folder/process.log for details" | tee -a $LOG_FILE
  fi
done

echo "All processing completed at $(date)" | tee -a $LOG_FILE

echo -e "\nProcessing summary:" | tee -a $LOG_FILE
for i in $(seq 1 $NUM_FOLDERS); do
  if [ -f "output/$i/video_out.mp4" ] && [ -s "output/$i/video_out.mp4" ]; then
    echo "Folder $i: SUCCESS" | tee -a $LOG_FILE
  elif [ -d "assets/$i" ] && [ -f "assets/$i/$i.mp4" ] && [ -f "assets/$i/output_stereo.mp3" ]; then
    echo "Folder $i: FAILED" | tee -a $LOG_FILE
  else
    echo "Folder $i: SKIPPED (files missing)" | tee -a $LOG_FILE
  fi
done