python main_image.py \
-b 10 \
-j 20 \
-lr 1e-2 \
--steps 50 \
--epochs 60 \
--mode only_tv \
--model_file save_model/admm_reconstruct.pth \
--psf_file data/psf.tiff \
--data_path /media/ausu-x299/diffuserCam_dataset/ \
--layer_num 10 \
--gpu 0  \
--log_file "save_model/admm_reconstruct.log" \
--filter_num 32 \
--kernel_size 3
