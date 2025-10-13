exp_name="V0_6_5_ET1_LPIPS20_05_k5_base10"
appearance_dim=0
visible_threshold=-1 #0.9
progressive="True"
resolution=-1
sh_degree=3
ET_final=1.0
lpips_interval=20
lambda_dlpips=0.5
watermark="True"
base_layer=10

# example:
./train.sh -d 'bungeenerf/amsterdam' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} \
   --lpips_interval ${lpips_interval} --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} \
   --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} --watermark ${watermark} \
   --base_layer ${base_layer} &
sleep 20s

./train.sh -d 'bungeenerf/barcelona' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} \
   --lpips_interval ${lpips_interval} --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} \
   --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} --watermark ${watermark} \
   --base_layer ${base_layer} &
sleep 1800s

./train.sh -d 'bungeenerf/bilbao' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} \
   --lpips_interval ${lpips_interval} --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} \
   --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} --watermark ${watermark} \
   --base_layer ${base_layer} &
sleep 20s

./train.sh -d 'bungeenerf/chicago' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} \
   --lpips_interval ${lpips_interval} --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} \
   --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} --watermark ${watermark} \
   --base_layer ${base_layer} &
sleep 1800s

./train.sh -d 'bungeenerf/hollywood' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} \
   --lpips_interval ${lpips_interval} --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} \
   --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} --watermark ${watermark} \
   --base_layer ${base_layer} &
sleep 20s

./train.sh -d 'bungeenerf/pompidou' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} \
   --lpips_interval ${lpips_interval} --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} \
   --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} --watermark ${watermark} \
   --base_layer ${base_layer}&
sleep 1800s

./train.sh -d 'bungeenerf/quebec' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} \
   --lpips_interval ${lpips_interval} --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} \
   --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} --watermark ${watermark} \
   --base_layer ${base_layer}&
sleep 20s

./train.sh -d 'bungeenerf/rome' -l ${exp_name} -r ${resolution} --appearance_dim ${appearance_dim} \
   --lpips_interval ${lpips_interval} --lambda_dlpips ${lambda_dlpips} --visible_threshold ${visible_threshold} \
   --progressive ${progressive} --sh_degree ${sh_degree} --ET_final ${ET_final} --watermark ${watermark} \
   --base_layer ${base_layer}&
