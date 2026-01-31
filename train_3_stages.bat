@echo off
setlocal enabledelayedexpansion

REM ===== FrameDiffuser Three-Stage Training =====
REM Stage 1: ControlNet only with black irradiance
REM Stage 2: ControlLoRA with temporal conditioning
REM Stage 3: Self-conditioning with generated frames

REM ===== CONFIGURATION - EDIT THESE =====
set BASE_MODEL=runwayml/stable-diffusion-v1-5
set DATA_DIR=./data/train
set VAL_DIR=./data/validation
set OUTPUT_DIR=./output
set RESOLUTION=512
set BATCH_SIZE=2
set GRADIENT_ACCUMULATION=4
set NUM_WORKERS=4
set TRAIN_PROMPT=A photorealistic scene

REM Training steps per stage
set STEPS_S1=40000
set STEPS_S2=10000
set STEPS_S3=30000

REM ========================================

echo.
echo ========================================
echo FrameDiffuser 3-Stage Training
echo ========================================
echo Model: %BASE_MODEL%
echo Data: %DATA_DIR%
echo Validation: %VAL_DIR%
echo Output: %OUTPUT_DIR%
echo.

REM ===== STAGE 1: CONTROLNET =====
if exist "%OUTPUT_DIR%\stage1\controlnet\diffusion_pytorch_model.safetensors" (
    echo Stage 1 checkpoint exists, skipping...
    goto STAGE2
)

echo.
echo === STAGE 1: ControlNet Only ===
echo.

python train.py ^
    --pretrained_model_name_or_path=%BASE_MODEL% ^
    --output_dir=%OUTPUT_DIR%/stage1 ^
    --gbuffer_root_dir=%DATA_DIR% ^
    --validation_folder=%VAL_DIR% ^
    --train_prompt="%TRAIN_PROMPT%" ^
    --resolution=%RESOLUTION% ^
    --train_batch_size=%BATCH_SIZE% ^
    --gradient_accumulation_steps=%GRADIENT_ACCUMULATION% ^
    --dataloader_num_workers=%NUM_WORKERS% ^
    --max_train_steps=%STEPS_S1% ^
    --checkpointing_steps=5000 ^
    --learning_rate=2e-5 ^
    --lr_scheduler=cosine ^
    --lr_warmup_steps=2000 ^
    --validation_steps=1000 ^
    --train_controlnet ^
    --use_black_irradiance ^
    --gradient_checkpointing ^
    --report_to=tensorboard ^
    --allow_tf32 ^
    --mixed_precision=bf16

if %ERRORLEVEL% neq 0 (
    echo Stage 1 FAILED
    pause
    exit /b 1
)
echo Stage 1 complete!

:STAGE2
REM ===== STAGE 2: CONTROLLORA =====
if exist "%OUTPUT_DIR%\stage2\pytorch_lora_weights.safetensors" (
    echo Stage 2 checkpoint exists, skipping...
    goto STAGE3
)

echo.
echo === STAGE 2: ControlLoRA Only ===
echo.

python train.py ^
    --pretrained_model_name_or_path=%BASE_MODEL% ^
    --output_dir=%OUTPUT_DIR%/stage2 ^
    --gbuffer_root_dir=%DATA_DIR% ^
    --validation_folder=%VAL_DIR% ^
    --train_prompt="%TRAIN_PROMPT%" ^
    --resolution=%RESOLUTION% ^
    --train_batch_size=%BATCH_SIZE% ^
    --gradient_accumulation_steps=%GRADIENT_ACCUMULATION% ^
    --dataloader_num_workers=%NUM_WORKERS% ^
    --max_train_steps=%STEPS_S2% ^
    --checkpointing_steps=2500 ^
    --learning_rate=2e-4 ^
    --lr_scheduler=cosine ^
    --lr_warmup_steps=1000 ^
    --validation_steps=500 ^
    --train_controllora ^
    --use_controlnet ^
    --controlnet_model_name_or_path=%OUTPUT_DIR%/stage1/controlnet ^
    --rank=64 ^
    --gradient_checkpointing ^
    --report_to=tensorboard ^
    --allow_tf32 ^
    --mixed_precision=bf16

if %ERRORLEVEL% neq 0 (
    echo Stage 2 FAILED
    pause
    exit /b 1
)
echo Stage 2 complete!

:STAGE3
REM ===== STAGE 3: SELF-CONDITIONING =====
if exist "%OUTPUT_DIR%\stage3\controlnet\diffusion_pytorch_model.safetensors" (
    if exist "%OUTPUT_DIR%\stage3\pytorch_lora_weights.safetensors" (
        echo Stage 3 complete, skipping...
        goto DONE
    )
)

echo.
echo === STAGE 3: Both + Self-Conditioning ===
echo.

python train.py ^
    --pretrained_model_name_or_path=%BASE_MODEL% ^
    --output_dir=%OUTPUT_DIR%/stage3 ^
    --gbuffer_root_dir=%DATA_DIR% ^
    --validation_folder=%VAL_DIR% ^
    --train_prompt="%TRAIN_PROMPT%" ^
    --resolution=%RESOLUTION% ^
    --train_batch_size=%BATCH_SIZE% ^
    --gradient_accumulation_steps=%GRADIENT_ACCUMULATION% ^
    --dataloader_num_workers=%NUM_WORKERS% ^
    --max_train_steps=%STEPS_S3% ^
    --checkpointing_steps=5000 ^
    --learning_rate=5e-5 ^
    --controlnet_learning_rate=2e-5 ^
    --lr_scheduler=cosine ^
    --lr_warmup_steps=1000 ^
    --validation_steps=1000 ^
    --train_controllora ^
    --train_controlnet ^
    --controlnet_model_name_or_path=%OUTPUT_DIR%/stage1/controlnet ^
    --load_lora_weights=%OUTPUT_DIR%/stage2 ^
    --rank=64 ^
    --use_generated_conditioning ^
    --generated_conditioning_freq=2000 ^
    --generated_conditioning_count=500 ^
    --generated_sample_ratio=0.5 ^
    --generated_conditioning_start_step=0 ^
    --generated_chaining_ratio=0.5 ^
    --gradient_checkpointing ^
    --report_to=tensorboard ^
    --allow_tf32 ^
    --mixed_precision=bf16

if %ERRORLEVEL% neq 0 (
    echo Stage 3 FAILED
    pause
    exit /b 1
)

:DONE
echo.
echo ========================================
echo TRAINING COMPLETE
echo ========================================
echo.
echo Models saved to:
echo   ControlNet: %OUTPUT_DIR%/stage3/controlnet
echo   ControlLoRA: %OUTPUT_DIR%/stage3/pytorch_lora_weights.safetensors
echo.

pause
