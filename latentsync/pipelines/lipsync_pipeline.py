# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py
import gc
import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess

import numpy as np
import torch
import torchvision

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_audio, write_video, check_ffmpeg_installed, read_video_batch, \
    combine_video_parts
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        denoising_unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(denoising_unet.config, "_diffusers_version") and version.parse(
            version.parse(denoising_unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = (
            hasattr(denoising_unet.config, "sample_size") and denoising_unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(denoising_unet.config)
            new_config["sample_size"] = 64
            denoising_unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.denoising_unet, "_hf_hook"):
            return self.device
        for module in self.denoising_unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_frames: np.ndarray):
        faces = []
        boxes = []
        affine_matrices = []
        face_indices = []
        face_detected_mask = []

        print(f"Affine transforming {len(video_frames)} frames...")
        for i, frame in enumerate(tqdm.tqdm(video_frames)):
            result = self.image_processor.affine_transform(frame)
            if result is not None:
                face, box, affine_matrix = result
                faces.append(face)
                boxes.append(box)
                affine_matrices.append(affine_matrix)
                face_indices.append(i)
                face_detected_mask.append(True)
            else:
                face_detected_mask.append(False)

        if not faces:
            raise RuntimeError("No faces detected in the entire video")

        faces = torch.stack(faces)
        return faces, boxes, affine_matrices, face_indices, face_detected_mask

    def affine_transform_video_safe(self, video_frames: np.ndarray, max_workers=8):
        from concurrent.futures import ThreadPoolExecutor

        def process_frame(frame):
            try:
                face, box, affine_matrix = self.image_processor.affine_transform_safe(frame)
                is_face_detected = box is not None and affine_matrix is not None
                return face, box, affine_matrix, is_face_detected
            except Exception as e:
                print(f"Error during affine transform: {e}")
                resized_frame = cv2.resize(frame, (self.image_processor.resolution, self.image_processor.resolution),
                                           interpolation=cv2.INTER_LANCZOS4)
                face_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1)
                return face_tensor, None, None, False

        print(f"Affine transforming {len(video_frames)} frames using {max_workers} threads...")
        faces = []
        boxes = []
        affine_matrices = []
        face_detected_mask = []

        # Используем map для сохранения порядка
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm.tqdm(executor.map(process_frame, video_frames), total=len(video_frames)))

        for face, box, affine_matrix, is_face_detected in results:
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)
            face_detected_mask.append(is_face_detected)

        faces = torch.stack(faces)
        face_detected_mask = np.array(face_detected_mask)

        return faces, boxes, affine_matrices, face_detected_mask

    def restore_single_frame(self, processed_frame, original_frame, box, affine_matrix):
        """
        Restores the processed frame to its original dimensions.

        Args:
            processed_frame: Processed frame (tensor).
            original_frame: Original frame.
            box: Face bounding box.
            affine_matrix: Affine transformation matrix.

        Returns:
            np.ndarray: Restored frame.
        """
        # If the face was not detected, we return the original frame
        if box is None or affine_matrix is None:
            return original_frame

        try:
            # Converting a tensor to a numpy array
            if isinstance(processed_frame, torch.Tensor):
                # Convert from (C, H, W) format to (H, W, C)
                x1, y1, x2, y2 = box
                height = int(y2 - y1)
                width = int(x2 - x1)

                # Resizing and formatting as in the original code
                processed_frame = torchvision.transforms.functional.resize(
                    processed_frame, size=(height, width), antialias=True
                )
                processed_frame = rearrange(processed_frame, "c h w -> h w c")
                processed_frame = (processed_frame / 2 + 0.5).clamp(0, 1)
                processed_frame = (processed_frame * 255).to(torch.uint8).cpu().numpy()

            # Using restorer.restore_img from image_processor
            out_frame = self.image_processor.restorer.restore_img(
                original_frame, processed_frame, affine_matrix
            )

            return out_frame
        except Exception as e:
            print(f"Error in restore_single_frame: {e}")
            print(f"Box: {box}")
            # Returning the original frame in case of an error
            return original_frame

    # def restore_video(self, faces, video_frames, boxes, affine_matrices):
    #     video_frames = video_frames[: faces.shape[0]]
    #     out_frames = []
    #     print(f"Restoring {len(faces)} faces...")
    #     for index, face in enumerate(tqdm.tqdm(faces)):
    #         x1, y1, x2, y2 = boxes[index]
    #         height = int(y2 - y1)
    #         width = int(x2 - x1)
    #         face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
    #         face = rearrange(face, "c h w -> h w c")
    #         face = (face / 2 + 0.5).clamp(0, 1)
    #         face = (face * 255).to(torch.uint8).cpu().numpy()
    #         # face = cv2.resize(face, (width, height), interpolation=cv2.INTER_LANCZOS4)
    #         out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
    #         out_frames.append(out_frame)
    #     return np.stack(out_frames, axis=0)

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 64,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask: str = "fix_mask",
        mask_image_path: str = "latentsync/utils/mask.png",
        batch_size: int = 1,
        max_batch_frames: int = 256,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.denoising_unet.training
        self.denoising_unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        batch_size = 1
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda", mask_image=mask_image)

        # 1. Default height and width to unet
        height = height or self.denoising_unet.config.sample_size * self.vae_scale_factor
        width = width or self.denoising_unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 5. Audio features preparing
        print("Extracting audio features...")
        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
        print(f"Audio chunks created for {video_fps} FPS")
        audio_samples = read_audio(audio_path)

        # Converting video to 25 FPS first
        temp_fps_dir = "temp_fps_conversion"
        if os.path.exists(temp_fps_dir):
            shutil.rmtree(temp_fps_dir)
        os.makedirs(temp_fps_dir, exist_ok=True)

        temp_video_path = os.path.join(temp_fps_dir, "video_25fps.mp4")
        command = f"ffmpeg -loglevel error -y -nostdin -i {video_path} -r 25 -c:v libx264 -crf 0 -preset veryslow -pix_fmt yuv444p {temp_video_path}"
        print(f"Converting video to 25 FPS: {command}")
        subprocess.run(command, shell=True)

        # Update video path for all subsequent operations and FPS for subsequent calculations
        video_path = temp_video_path
        video_fps = 25

        #6. Getting information about the video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Creating a temporary directory
        temp_dir = "temp_processing"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        print(f"Converted video info: {total_frames} frames at {video_fps} FPS")

        total_frames = min(total_frames, len(whisper_chunks))
        print(f"Will process {total_frames} frames")

        # Determining the batch size (multiple of num_frames)
        batch_frames = min(num_frames * max(1, max_batch_frames // num_frames), max_batch_frames)
        print(f"Processing video in batches of {batch_frames} frames")

        # Preparation for processing
        processed_frames = 0
        batch_count = 0
        part_paths = []

        # Number of channels for latent variables
        num_channels_latents = self.vae.config.latent_channels

        # Progress bar for batches
        total_batches = math.ceil(total_frames / batch_frames)
        batch_progress = tqdm.tqdm(total=total_batches, desc="Processing video batches")

        #7. Video processing in parts
        while processed_frames < total_frames:
            # Clearing CUDA memory before processing a new batch
            torch.cuda.empty_cache()

            print(
                f"Debug: processed_frames = {processed_frames}, total_frames = {total_frames}, batch_frames = {batch_frames}")

            # Defining the boundaries of the current batch
            start_frame = processed_frames
            end_frame = min(processed_frames + batch_frames, total_frames)
            current_batch_size = end_frame - start_frame

            print(
                f"Debug: start_frame = {start_frame}, end_frame = {end_frame}, current_batch_size = {current_batch_size}")

            # Reading from the converted video
            video_frames_batch = read_video_batch(video_path, start_frame, end_frame)

            print(f"Debug: read_video_batch returned {len(video_frames_batch)} frames")

            # Check that we have received the frames
            if len(video_frames_batch) == 0:
                print(f"Warning: read_video_batch returned 0 frames for range {start_frame}-{end_frame}")
                processed_frames = end_frame
                batch_count += 1
                continue

            # Processing faces in the batch
            print(f"Processing faces in batch {batch_count + 1}/{total_batches}")
            faces_batch, boxes_batch, affine_matrices_batch, face_detected_mask_batch = self.affine_transform_video_safe(
                video_frames_batch)

            # Determining the number of groups by num_frames of frames for inference
            num_inferences_batch = current_batch_size // num_frames
            if current_batch_size % num_frames != 0:
                # Adding another group for the remaining frames
                num_inferences_batch += 1

            synced_video_frames_batch = []

            # Preparing latent variables for the entire batch
            all_latents = self.prepare_latents(
                batch_size,
                current_batch_size,
                num_channels_latents,
                height,
                width,
                weight_dtype,
                device,
                generator,
            )

            #8. Processing each group of frames in the batch
            for i in range(num_inferences_batch):
                start_idx = i * num_frames
                end_idx = min((i + 1) * num_frames, current_batch_size)

                # Handling incomplete groups by padding
                if end_idx - start_idx < num_frames:
                    print(
                        f"Incomplete group of {end_idx - start_idx} frames (less than {num_frames}), padding to full size")

                    # Create padded versions of the data
                    padding_needed = num_frames - (end_idx - start_idx)

                    # Pad video frames by duplicating the last frame
                    last_frame = video_frames_batch[end_idx - 1]
                    padded_frames = np.array([video_frames_batch[j] for j in range(start_idx, end_idx)])
                    for _ in range(padding_needed):
                        padded_frames = np.append(padded_frames, [last_frame], axis=0)

                    # Pad faces
                    last_face = faces_batch[end_idx - 1]
                    padded_faces = faces_batch[start_idx:end_idx].clone()
                    padding_faces = last_face.unsqueeze(0).repeat(padding_needed, 1, 1, 1)
                    padded_faces = torch.cat([padded_faces, padding_faces], dim=0)

                    # Pad face detection mask
                    padded_mask = np.append(face_detected_mask_batch[start_idx:end_idx],
                                            [face_detected_mask_batch[end_idx - 1]] * padding_needed)

                    # Check if there are faces in the padded group
                    if not any(padded_mask):
                        print(
                            f"  No faces detected in frames {start_idx + start_frame} to {end_idx + start_frame}, using original frames")
                        original_frames = [torch.from_numpy(frame).permute(2, 0, 1) for frame in
                                           video_frames_batch[start_idx:end_idx]]
                        synced_video_frames_batch.extend(original_frames)
                        continue

                    # Use the padded data for processing
                    current_faces = padded_faces

                    # Making sure we have enough audio embeddings
                    current_audio_embeds = []
                    for j in range(end_idx - start_idx + padding_needed):
                        idx = start_frame + start_idx + j
                        if j < end_idx - start_idx:
                            if idx < len(whisper_chunks):
                                current_audio_embeds.append(whisper_chunks[idx])
                            else:
                                # If there are not enough audio chunks, duplicate the last one
                                print(f"Warning: Using duplicate audio chunk for frame {idx}")
                                current_audio_embeds.append(whisper_chunks[-1])
                        else:
                            # Padding audio with duplicates of last frame
                            current_audio_embeds.append(current_audio_embeds[-1])

                    current_audio_embeds = torch.stack(current_audio_embeds)
                    current_audio_embeds = current_audio_embeds.to(device, dtype=weight_dtype)

                    if do_classifier_free_guidance:
                        null_audio_embeds = torch.zeros_like(current_audio_embeds)
                        current_audio_embeds = torch.cat([null_audio_embeds, current_audio_embeds])

                    # Extracting latent variables for the padded group
                    latents = all_latents[:, :, start_idx:end_idx]

                    # Padding latents
                    last_frame_latent = latents[:, :, -1:]
                    padding_latents = torch.cat([last_frame_latent] * padding_needed,
                                                dim=2)
                    latents = torch.cat([latents, padding_latents], dim=2)

                    # Preparing masks and images with masks
                    ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                        current_faces, affine_transform=False
                    )

                    # Preparing latent variables for masks
                    mask_latents, masked_image_latents = self.prepare_mask_latents(
                        masks,
                        masked_pixel_values,
                        height,
                        width,
                        weight_dtype,
                        device,
                        generator,
                        do_classifier_free_guidance,
                    )

                    # Preparing latent variables for images
                    ref_latents = self.prepare_image_latents(
                        ref_pixel_values,
                        device,
                        weight_dtype,
                        generator,
                        do_classifier_free_guidance,
                    )

                    # The denoising cycle
                    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                    with self.progress_bar(total=num_inference_steps) as progress_bar:
                        for j, t in enumerate(timesteps):
                            denoising_unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                            denoising_unet_input = self.scheduler.scale_model_input(denoising_unet_input, t)

                            denoising_unet_input = torch.cat(
                                [denoising_unet_input, mask_latents, masked_image_latents, ref_latents], dim=1
                            )

                            noise_pred = self.denoising_unet(
                                denoising_unet_input, t, encoder_hidden_states=current_audio_embeds
                            ).sample

                            if do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                            if j == len(timesteps) - 1 or (
                                    (j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                                progress_bar.update()
                                if callback is not None and j % callback_steps == 0:
                                    callback(j, t, latents)

                    # Decoding latent variables into images
                    decoded_latents = self.decode_latents(latents)

                    # Inserting the surrounding pixels back in
                    decoded_latents = self.paste_surrounding_pixels_back(
                        decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
                    )

                    # Take only the original frames (discard padded results)
                    original_count = end_idx - start_idx
                    for k in range(original_count):
                        is_face_detected = face_detected_mask_batch[start_idx + k]
                        if is_face_detected:
                            synced_video_frames_batch.append(decoded_latents[k])
                        else:
                            orig_frame = torch.from_numpy(video_frames_batch[start_idx + k]).permute(2, 0, 1)
                            synced_video_frames_batch.append(orig_frame)

                    # Clearing the memory after processing the group
                    del latents, mask_latents, masked_image_latents, ref_latents, decoded_latents
                    torch.cuda.empty_cache()

                    continue

                # Check if there are faces in the current group of frames
                if not any(face_detected_mask_batch[start_idx:end_idx]):
                    print(
                        f"  No faces detected in frames {start_idx + start_frame} to {end_idx + start_frame}, skipping processing")
                    # Adding original frames
                    original_frames = [torch.from_numpy(frame).permute(2, 0, 1) for frame in
                                       video_frames_batch[start_idx:end_idx]]
                    synced_video_frames_batch.extend(original_frames)
                    continue

                # Extracting data for the current group
                current_faces = faces_batch[start_idx:end_idx]

                # Making sure we have enough audio embeddings
                current_audio_embeds = []
                for idx in range(start_frame + start_idx, start_frame + end_idx):
                    if idx < len(whisper_chunks):
                        current_audio_embeds.append(whisper_chunks[idx])
                    else:
                        # If there are not enough audio chunks, duplicate the last one
                        print(f"Warning: Using duplicate audio chunk for frame {idx}")
                        current_audio_embeds.append(whisper_chunks[-1])

                current_audio_embeds = torch.stack(current_audio_embeds)
                current_audio_embeds = current_audio_embeds.to(device, dtype=weight_dtype)

                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(current_audio_embeds)
                    current_audio_embeds = torch.cat([null_audio_embeds, current_audio_embeds])

                # Extracting latent variables for the current group
                latents = all_latents[:, :, start_idx:end_idx]

                # Preparing masks and images with masks
                ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                    current_faces, affine_transform=False
                )

                # Preparing latent variables for masks
                mask_latents, masked_image_latents = self.prepare_mask_latents(
                    masks,
                    masked_pixel_values,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                    do_classifier_free_guidance,
                )

                # Preparing latent variables for images
                ref_latents = self.prepare_image_latents(
                    ref_pixel_values,
                    device,
                    weight_dtype,
                    generator,
                    do_classifier_free_guidance,
                )

                # The denoising cycle
                num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for j, t in enumerate(timesteps):
                        denoising_unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                        denoising_unet_input = self.scheduler.scale_model_input(denoising_unet_input, t)

                        denoising_unet_input = torch.cat(
                            [denoising_unet_input, mask_latents, masked_image_latents, ref_latents], dim=1
                        )

                        noise_pred = self.denoising_unet(
                            denoising_unet_input, t, encoder_hidden_states=current_audio_embeds
                        ).sample

                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                        if j == len(timesteps) - 1 or (
                                (j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                            if callback is not None and j % callback_steps == 0:
                                callback(j, t, latents)

                # Decoding latent variables into images
                decoded_latents = self.decode_latents(latents)

                # Inserting the surrounding pixels back in
                decoded_latents = self.paste_surrounding_pixels_back(
                    decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
                )

                # Adding processed frames to the result
                current_faces_detected = face_detected_mask_batch[start_idx:end_idx]

                # Processing the results for each frame
                for k, is_face_detected in enumerate(current_faces_detected):
                    if k < len(decoded_latents):
                        # If a face is detected, we use the processed frame
                        if is_face_detected:
                            synced_video_frames_batch.append(decoded_latents[k])
                        else:
                            # If the face is not detected, we use the original frame
                            orig_frame = torch.from_numpy(video_frames_batch[start_idx + k]).permute(2, 0, 1)
                            synced_video_frames_batch.append(orig_frame)

                # Clearing the memory after processing the group
                del latents, mask_latents, masked_image_latents, ref_latents, decoded_latents
                torch.cuda.empty_cache()

            # Restoring the full video for the current batch
            restored_frames = []
            for i, frame in enumerate(synced_video_frames_batch):
                if face_detected_mask_batch[i]:
                    restored_frame = self.restore_single_frame(
                        frame, video_frames_batch[i], boxes_batch[i], affine_matrices_batch[i]
                    )
                    restored_frames.append(restored_frame)
                else:
                    restored_frames.append(video_frames_batch[i])

            batch_output_path = os.path.join(temp_dir, f"batch_{batch_count:04d}.mp4")
            write_video(batch_output_path, np.array(restored_frames), fps=video_fps)
            part_paths.append(batch_output_path)

            del video_frames_batch, faces_batch, boxes_batch, affine_matrices_batch
            del synced_video_frames_batch, restored_frames, all_latents
            gc.collect()
            torch.cuda.empty_cache()

            batch_count += 1
            batch_progress.update(1)

            print(f"Debug: Completing batch {batch_count}, processed_frames was {processed_frames}")
            processed_frames += current_batch_size
            print(f"Debug: processed_frames now {processed_frames}")

        batch_progress.close()

        # 9. Combine all the parts of the processed video
        print("Combining processed video parts...")
        combined_video_path = os.path.join(temp_dir, "video.mp4")
        combine_video_parts(part_paths, combined_video_path)

        # 10. Trim the audio to the length of the video
        print("Processing audio...")
        cap = cv2.VideoCapture(combined_video_path)
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        audio_samples_remain_length = int(video_frame_count / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()
        audio_output_path = os.path.join(temp_dir, "audio.wav")
        sf.write(audio_output_path, audio_samples, audio_sample_rate)

        # 11. Combining video and audio
        print("Combining video and audio...")
        command = f"ffmpeg -y -loglevel error -i {combined_video_path} -i {audio_output_path} -c:v copy -c:a aac -b:a 320k -map 0:v:0 -map 1:a:0 {video_out_path}"
        subprocess.run(command, shell=True)

        # 12. Cleaning temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        if os.path.exists(temp_fps_dir):
            shutil.rmtree(temp_fps_dir)

        # Returning the model to its original mode
        if is_train:
            self.denoising_unet.train()

        print(f"Video processing completed: {video_out_path}")
        return video_out_path
