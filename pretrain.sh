method='commit'


# R2R
if [ $1 = "R2R" ]; then
    echo "Pretrain on R2R."
    python src/main.py --log \
        --method $method \
        --description 'pretrain,R2R,MLM,CLIP,KV' \
        --mask_visited \
        --use_panorama \
        --use_directed \
        --OPE_layer_num 2 \
        --MAM_layer_num 4 \
        --CCM_layer_num 1 \
        --text_backbone 'ALBEF' \
        --gpu '3' \
        --dataset 'R2RPretrain' \
        --model 'PRET_KV' \
        --agent 'AgentPretrain' \
        --trainer 'Pretrain' \
        --tasks MLM \
        --lr 0.00002 \
        --dropout 0.1 \
        --batch_size 16 \
        --lr_scheduler cosine \
        --iteration_num 100000 \
        --log_every 1000
fi


# RxR
# Note that text encoder is replaced with the multilingual XLMRoBERTa
if [ $1 = "RxR" ]; then
    echo "Pretrain on RxR."
    python src/main.py --log \
        --method $method \
        --description 'pretrain,RxR,MLM,CLIP' \
        --mask_visited \
        --use_panorama \
        --use_directed \
        --OPE_layer_num 2 \
        --MAM_layer_num 4 \
        --CCM_layer_num 1 \
        --dataset 'RxRPretrain' \
        --model 'PRET' \
        --agent 'AgentPretrain' \
        --trainer 'Pretrain' \
        --tasks MLM \
        --gpu '0' \
        --lr 0.00002 \
        --dropout 0.1 \
        --batch_size 16 \
        --lr_scheduler cosine \
        --iteration_num 100000 \
        --log_every 5000
fi
