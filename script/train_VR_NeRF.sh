
python train.py --eval -s data/VR_NeRF/apartment -m experiments/VR_NeRF/apartment/base6 --max_sh_degree 3 \
--lpips_interval 20 --dense_score --dense_score_threshold 999.0 --densify_grad_threshold 0.0002 --depth_correct \
--ET_grade 2.0 --ET_grade_final 1.0 --iterations 100_000 -r -1 --lambda_dlpips 0.5 --appearance_dim 0 \
--progressive --update_until 50_000 --data_device disk --base_layer 6 &
sleep 30s

python train.py --eval -s data/VR_NeRF/kitchen -m experiments/VR_NeRF/kitchen/base7 --max_sh_degree 3 \
--lpips_interval 20 --dense_score --dense_score_threshold 999.0 --densify_grad_threshold 0.0002 --depth_correct \
--ET_grade 2.0 --ET_grade_final 1.0 --iterations 100_000 -r -1 --lambda_dlpips 0.5 --appearance_dim 0 \
--progressive --update_until 50_000 --data_device disk --base_layer 7 &

