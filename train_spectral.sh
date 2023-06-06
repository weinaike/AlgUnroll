python main_spectral.py \
-b 4 \
-j 4 \
-lr 1e-2 \
--steps 50 \
--epochs 60 \
--mode tv_cnn \
--model_file save_model/spectral_fc_01.pth \
--sp_file data/SpectralResponse_9.npy \
--data_path data/SpectralResponse_9_1024_multi \
--layer_num 50 \
--gpu 0 \
--log_file "save_model/spectral_admm.log" \
--filter_num 32 \
--kernel_size 3
