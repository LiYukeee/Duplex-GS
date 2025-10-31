#!/bin/bash

# training settings
record_running_output=true
iterations=40000
warmup=false
progressive=true
# data path, scenes and output path
DATA_BASE_PATH="data/mipnerf360"
OUTPUT_BASE_PATH="outputs/mipnerf360"
scenes=(
  "bicycle"
  "bonsai"
  "counter"
  "flowers"
  "garden"
  "kitchen"
  "room"
  "stump"
  "treehill"
)

# create command list
command_list=()
for scene in ${scenes[@]}; do
    # input and output paths
    scene_output_path=${OUTPUT_BASE_PATH}/${scene}
    data_path=${DATA_BASE_PATH}/${scene}
    mkdir -p ${scene_output_path}

    ### train command ###
    extra_args=" --eval --iterations ${iterations} --max_sh_degree 3 --lpips_interval 20 --dense_score --dense_score_threshold 999.0 --densify_grad_threshold 0.0002 --depth_correct --anchor_search --ET_grade 2.0 --ET_grade_final 1.0 -r -1 --base_layer -1 --lambda_dlpips 0.5 --appearance_dim 0 "
    [ "$progressive" = true ] && extra_args+=" --progressive "
    [ "$warmup" = true ] && extra_args+=" --warmup "
    [ "$record_running_output" = true ] && extra_args+=" &>> ${scene_output_path}/running.txt "
    command="python train.py -s ${data_path} -m ${scene_output_path} ${extra_args}"
    echo execute command: $command
    command_list+=("$command")

    # ## render command ###
    # extra_args=" --render_video --skip_train --skip_test --data_device disk"
    # [ "$record_running_output" = true ] && extra_args+=" &>> ${scene_output_path}/running.txt"
    # command="python render.py -m ${scene_output_path} ${extra_args}"
    # echo execute command: $command
    # command_list+=("$command")
    
done

# parallel execution
n_jobs=4
delay_time=20
parallel --jobs ${n_jobs} --delay ${delay_time} ::: "${command_list[@]}"