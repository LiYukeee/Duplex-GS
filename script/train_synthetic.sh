exp_name="V0_6_5_ET1_LPIPS20_05_k5"
appearance_dim=0
visible_threshold=-1 #0.9
progressive="True"
resolution=-1
sh_degree=3
ET_final=1.0
lpips_interval=20
lambda_dlpips=0.5

./train.sh -d 'nerf_synthetic/chair' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 20s

./train.sh -d 'nerf_synthetic/drums' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 20s

./train.sh -d 'nerf_synthetic/ficus' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 20s

./train.sh -d 'nerf_synthetic/hotdog' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 1800s

./train.sh -d 'nerf_synthetic/lego' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 20s

./train.sh -d 'nerf_synthetic/materials' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 20s

./train.sh -d 'nerf_synthetic/mic' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 20s

./train.sh -d 'nerf_synthetic/ship' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 20s
