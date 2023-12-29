exp_name=zero_shot_facevc
inference_gpu=0

config_file=./output/${exp_name}/config.ini

python Tools/modify_config.py --config_file $config_file --inference_gpu $inference_gpu --output_root $output_root


python inference.py --config_file $config_file || exit 