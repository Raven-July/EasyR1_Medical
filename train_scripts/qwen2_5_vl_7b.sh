set -x

MODEL_PATH=/groups/xitucheng213/home/share/yzy/Pretrained/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

SYSTEM_PROMPT="""This is a medical image classification task, the final answer MUST BE put in \\boxed{}."""

python3 -m verl.trainer.main \
    config=/groups/xitucheng213/home/share/yzy/Vision-R1-Projects/EasyR1/train_scripts/config.yaml \
    data.train_files=/groups/xitucheng213/home/share/yzy/Vision-R1-Projects/Datasets/MedMNIST/converted/BreastMNIST-256@train \
    data.val_files=/groups/xitucheng213/home/share/yzy/Vision-R1-Projects/Datasets/MedMNIST/converted/BreastMNIST-256@test \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.rollout_batch_size=256 \
    data.val_batch_size=-1 \
    data.data_type=arrow \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.reward.compute_score=math \
    trainer.experiment_name=qwen2_5_vl_7b_BreastMNIST+original-256 \
    trainer.logger=['console','swanlab'] \
    trainer.n_gpus_per_node=2 \
    trainer.val_generations_to_log=10 \
    trainer.total_episodes=0 \
    trainer.val_freq=5 \
    trainer.save_freq=30 \
    trainer.save_limit=2 \
    trainer.save_checkpoint_path=/groups/xitucheng213/home/share/yzy/Vision-R1-Projects/EasyR1/checkpoints/MedMNIST/OrganAMNIST+original-256
