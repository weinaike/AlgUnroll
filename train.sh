python main_image.py \
-b 16 \
-j 20 \
-lr 1e-3 \
--steps 20 \
--epochs 30 \
--mode only_tv \
--model_file debug/admm_reconstruct.pth \
--psf_file data/psf.tiff \
--data_path /media/ausu-x299/diffuserCam_dataset/ \
--layer_num 5 \
--gpu 0 \
--log_file "debug/admm_reconstruct.log" \
--filter_num 36 \
--kernel_size 3
