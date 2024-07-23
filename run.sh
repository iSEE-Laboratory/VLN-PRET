# NOTE: remember to change feature path in config.py if you change the feature

method='commit'

# R2R
if [ $1 = "R2R" ]; then
    echo "Training on R2R."
    # --not_load_strict \
    # --evaluate_first \
    python src/main.py --log \
        --method $method \
        --description 'R2R,MLM,CLIP,KV,' \
        --mask_visited \
        --use_panorama \
        --use_directed \
        --OPE_layer_num 2 \
        --MAM_layer_num 4 \
        --CCM_layer_num 1 \
        --dataset 'R2R' \
        --model 'PRET_KV' \
        --agent 'AgentPath_KV' \
        --trainer 'TF_SF' \
        --text_backbone 'ALBEF' \
        --gpu '1' \
        --lr 0.00001 \
        --batch_size 8 \
        --dropout 0.5 \
        --lr_scheduler cosine \
        --iteration_num 100000 \
        --log_every 1000 \
        --loss_weight 0.2 \
        --max_step 15 \
        --load 'temp/log/commit/2024-02-27_21:50:24_pretrain,R2R,MLM,CLIP'
        # --load 'temp/log/commit/2024-03-01_15:03:33_R2R,MLM,CLIP'
fi


# RxR dataset
# RxR requires more GPU memory, and longer max_step
# Gradient Checkpoint is used in multilingual Roberta to save memory
# RxR uses higher loss_weight to follow the path.
if [ $1 = "RxR" ]; then
    echo "Training on RxR."
    python src/main.py --log \
        --method $method \
        --description 'RxR,bs4,MLM,clip,' \
        --mask_visited \
        --use_panorama \
        --use_directed \
        --evaluate_first \
        --OPE_layer_num 2 \
        --MAM_layer_num 4 \
        --CCM_layer_num 1 \
        --dataset 'RxR' \
        --model 'PRET' \
        --agent 'AgentPath' \
        --trainer 'TF_SF' \
        --gpu '0' \
        --lr 0.00001 \
        --dropout 0.5 \
        --batch_size 4 \
        --lr_scheduler cosine \
        --iteration_num 200000 \
        --log_every 5000 \
        --loss_weight 0.4 \
        --max_step 20 \
        --load 'log/commit/2024-03-02_00:12:15_RxR,bs4,MLM,CLIP'
fi
