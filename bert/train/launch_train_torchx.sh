run_name="beauty_toy_sport_lr_1e-5_test_run"
name="${run_name}_BERT_train"
app_id='${app_id}'
torchx run \
    --scheduler_args="fbpkg_ids=torchx_conda_mount:stable,manifold.manifoldfs:prod"  \
    fb.conda.torchrun \
    --h zionex_80g \
    --run_as_root True \
    --env "DISABLE_NFS=1;DISABLE_OILFS=1;MANIFOLDFS_BUCKET=coin;LD_PRELOAD=/usr/local/fbcode/platform010/lib/libcuda.so:/usr/local/fbcode/platform010/lib/libnvidia-ml.so" \
    --name $name \
    -- \
     --no-python --nnodes=1 --nproc-per-node=8 \
    ./run.sh ./bertclassifier_train.py  \
    --output_dir="/mnt/mffuse/pretrain_recommendation/bertclassifier/${run_name}/" \
    --tensorboard_dir="/mnt/mffuse/pretrain_recommendation/bertclassifier/${run_name}/tensorboardlog/"
