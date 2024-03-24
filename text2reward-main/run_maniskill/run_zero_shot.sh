python ppo.py --env_id LiftCube-v0 \
    --train_num 8 --eval_num 5 --eval_freq 12800 --max_episode_steps 100 \
    --rollout_steps 3200 --train_max_steps 2_000_000 --seed 0 --eval_seed 1 \
    --reward_path ./reward_code/zero-shot/LiftCube-v0/specific.py --exp_name zero-shot

python ppo.py --env_id PickCube-v0 \
    --train_num 8 --eval_num 5 --eval_freq 12800 --max_episode_steps 100 \
    --rollout_steps 3200 --train_max_steps 4_000_000 --seed 0 --eval_seed 1 \
    --reward_path ./reward_code/zero-shot/PickCube-v0/specific.py --exp_name zero-shot

python sac.py --env_id TurnFaucet-v0 \
    --train_num 8 --eval_num 5 --eval_freq 16_000 --max_episode_steps 200 \
    --train_max_steps 4_000_000 --seed 0 --eval_seed 1 \
    --reward_path ./reward_code/zero-shot/TurnFaucet-v0/specific.py --exp_name zero-shot

python sac.py --env_id OpenCabinetDrawer-v1 \
    --train_num 8 --eval_num 5 --eval_freq 16_000 --max_episode_steps 200 \
    --train_max_steps 4_000_000 --seed 0 --eval_seed 1 \
    --reward_path ./reward_code/zero-shot/OpenCabinetDrawer-v1/specific.py --exp_name zero-shot

python sac.py --env_id OpenCabinetDoor-v1 \
    --train_num 8 --eval_num 5 --eval_freq 16_000 --max_episode_steps 200 \
    --train_max_steps 8_000_000 --seed 0 --eval_seed 1 \
    --reward_path ./reward_code/zero-shot/OpenCabinetDoor-v1/specific.py --exp_name zero-shot

python sac.py --env_id PushChair-v1 \
    --train_num 8 --eval_num 5 --eval_freq 16_000 --max_episode_steps 200 \
    --train_max_steps 8_000_000 --seed 0 --eval_seed 1 \
    --reward_path ./reward_code/zero-shot/PushChair-v1/specific.py --exp_name zero-shot