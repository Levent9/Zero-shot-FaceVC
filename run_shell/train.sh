exp_name=zero_shot_facevc
config_file=${exp_name}.ini
log_dir=./output/${exp_name}/log

mkdir -p $log_dir
echo $exp_name

# python train.py --config_file $config_file

nohup python -u train.py --config_file $config_file > $log_dir/exp.log 2>&1 &
tail -f ${log_dir}/exp.log