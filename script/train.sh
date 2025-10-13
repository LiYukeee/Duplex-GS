set -x
function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

port=$(rand 10000 30000)

iterations=40_000
base_layer=-1
warmup="False"
progressive="False"
watermark="False"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--logdir) logdir="$2"; shift ;;
        -r|--resolution) resolution="$2"; shift ;;
        -d|--data) data="$2"; shift ;;
        --ET_final) ET_final="$2"; shift ;;
        --sh_degree) sh_degree="$2"; shift ;;
        --base_layer) base_layer="$2"; shift ;;
        --appearance_dim) appearance_dim="$2"; shift ;;
        --visible_threshold ) visible_threshold="$2"; shift ;;
        --progressive) progressive="$2"; shift ;;
        --lpips_interval) lpips_interval="$2"; shift ;;
        --lambda_dlpips) lambda_dlpips="$2"; shift ;;
        --watermark) watermark="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

time=$(date "+%Y-%m-%d_%H:%M:%S")

if [ "$progressive" = "True" ]; then
  if [ "$watermark" = "True" ]; then
    mkdir -p ../experiments/${data}/${logdir}/$time
    python ../train.py --eval -s ../data/${data} -m ../experiments/${data}/${logdir}/$time --max_sh_degree ${sh_degree} --watermark \
    --lpips_interval ${lpips_interval} --dense_score --dense_score_threshold 999.0 --densify_grad_threshold 0.0002 \
    --depth_correct --anchor_search --ET_grade 2.0 --ET_grade_final ${ET_final} --iterations 40000 -r ${resolution} --base_layer ${base_layer} \
    --lambda_dlpips ${lambda_dlpips} --appearance_dim ${appearance_dim} --progressive &>> ../experiments/${data}/${logdir}/$time/run.txt &
  else
    mkdir -p ../experiments/${data}/${logdir}/$time
    python ../train.py --eval -s ../data/${data} -m ../experiments/${data}/${logdir}/$time --max_sh_degree ${sh_degree} \
    --lpips_interval ${lpips_interval} --dense_score --dense_score_threshold 999.0 --densify_grad_threshold 0.0002 \
    --depth_correct --anchor_search --ET_grade 2.0 --ET_grade_final ${ET_final} --iterations 40000 -r ${resolution} --base_layer ${base_layer} \
    --lambda_dlpips ${lambda_dlpips} --appearance_dim ${appearance_dim} --progressive &>> ../experiments/${data}/${logdir}/$time/run.txt &
  fi
else
  if [ "$watermark" = "True" ]; then
    mkdir -p ../experiments/${data}/${logdir}/$time
    python ../train.py --eval -s ../data/${data} -m ../experiments/${data}/${logdir}/$time --max_sh_degree ${sh_degree} --watermark \
    --lpips_interval ${lpips_interval} --dense_score --dense_score_threshold 999.0 --densify_grad_threshold 0.0002 \
    --depth_correct --anchor_search --ET_grade 2.0 --ET_grade_final 2.0 --iterations 40000 -r ${resolution} --base_layer ${base_layer} \
    --lambda_dlpips ${lambda_dlpips} --appearance_dim ${appearance_dim} &>> ../experiments/${data}/${logdir}/$time/run.txt &
  else
    mkdir -p ../experiments/${data}/${logdir}/$time
    python ../train.py --eval -s ../data/${data} -m ../experiments/${data}/${logdir}/$time --max_sh_degree ${sh_degree} \
    --lpips_interval ${lpips_interval} --dense_score --dense_score_threshold 999.0 --densify_grad_threshold 0.0002 \
    --depth_correct --anchor_search --ET_grade 2.0 --ET_grade_final 2.0 --iterations 40000 -r ${resolution} --base_layer ${base_layer} \
    --lambda_dlpips ${lambda_dlpips} --appearance_dim ${appearance_dim} &>> ../experiments/${data}/${logdir}/$time/run.txt &
  fi
fi
