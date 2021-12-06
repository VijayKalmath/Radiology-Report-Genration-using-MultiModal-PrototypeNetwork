python visualize.py \
    --image_dir data/iu_xray/images/ \
    --ann_path data/iu_xray/annotation.json \
    --label_path data/iu_xray/labels_14.pickle \
    --img_init_protypes_path data/iu_xray/init_protypes_2048_empty_224_both.pt \
    --text_init_protypes_path data/iu_xray/text_empty_initprotypes_512.pt \
    --init_protypes_path data/iu_xray/cat_initprotypes_2048_768_14_cluster10.pt \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --epochs 30 \
    --batch_size 4 \
    --lr_ve 1e-3 \
    --lr_ed 2e-3 \
    --step_size 10 \
    --gamma 0.8 \
    --num_layers 3 \
    --topk 15 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --seed 7580 \
    --beam_size 3 \
    --save_dir results/iu_xray/ \
    --log_period 50 \
    --n_gpu 1 \
    --num_cluster 14 \
    --img_num_protype 8 \
    --text_num_protype 4 \
    --gbl_num_protype 8 \
    --img_con_margin 0.4 \
    --txt_con_margin 0.4 \
    --weight_img_bce_loss 0 \
    --weight_txt_bce_loss 0 \
    --weight_txt_con_loss 1 \
    --weight_img_con_loss 1 \
    --d_img_ebd 2048 \
    --d_txt_ebd 768 \
    --num_protype 20

