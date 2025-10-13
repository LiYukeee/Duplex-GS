exp_name="V0_6_5_ET1_LPIPS20_05_k5_ablation_wo+anchor_search"
appearance_dim=0
visible_threshold=-1 #0.9
progressive="True"
resolution=-1
sh_degree=3
ET_final=1.0
lpips_interval=20
lambda_dlpips=0.5

./train.sh -d 'mipnerf360_1600/bicycle' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 20s

./train.sh -d 'mipnerf360_1600/garden' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 20s

./train.sh -d 'mipnerf360_1600/stump' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 3600s

./train.sh -d 'mipnerf360_1600/room' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 20s

./train.sh -d 'mipnerf360_1600/counter' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 20s

./train.sh -d 'mipnerf360_1600/kitchen' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 3600s

./train.sh -d 'mipnerf360_1600/bonsai' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 20s

./train.sh -d 'mipnerf360_1600/flowers' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 20s

./train.sh -d 'mipnerf360_1600/treehill' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} --lpips_interval ${lpips_interval} \
   --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} &
sleep 20s
