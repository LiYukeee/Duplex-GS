exp_name="V0_6_5_ET1_LPIPS20_05_k5_base10"
appearance_dim=0
visible_threshold=-1 #0.9
progressive="True"
resolution=-1
sh_degree=3
ET_final=1.0
lpips_interval=20
lambda_dlpips=0.2
base_layer=10

# example:
./train.sh -d 'tandt/truck' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} \
   --lpips_interval ${lpips_interval} --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} \
   --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} --base_layer ${base_layer} &
#sleep 20s

#./train.sh -d 'tandt/train' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} \
#   --lpips_interval ${lpips_interval} --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} \
#   --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} --base_layer ${base_layer} &
