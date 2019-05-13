
#!/bin/bash

mkdir log
mkdir ckpt

# mnist: batch_size=1000, lr=0.5, 
seed=${6}
dataset=cifar10
model=cnn
gsr=${1}
egsr=${2}
batch_size=256
lr=0.1
ssp=${5}

epoch_n=${3}


echo 'dataset: '$dataset' model: '$model' gsr: '$gsr' seed: '$seed

subfix=orig
mkdir -p ckpt/${dataset}/${ssp}_${gsr}/orig
mkdir -p log/${dataset}/${ssp}_${gsr}/orig

ckptdir=ckpt/${dataset}/${ssp}_${gsr}/orig
logdir=log/${dataset}/${ssp}_${gsr}/orig

python cifar_mnist_noisy.py --batch_size ${batch_size} --epoch_num ${epoch_n} --seed $seed \
        --debug 'no' --good_sample_ratio $gsr --sub_sampling $ssp --dataset ${dataset}  --model ${model} \
        --learning_rate ${lr} --log_file ${logdir}/${dataset}'_'$model'_gsr_'$gsr'_seed_'$seed'_ssp_'$ssp'_'${subfix}'.log' \
        --subfix ${subfix} --saveckptpath $ckptdir

mkdir -p ckpt/${dataset}/${ssp}_${gsr}/iter1
mkdir -p log/${dataset}/${ssp}_${gsr}/iter1

pckptdir=ckpt/${dataset}/${ssp}_${gsr}/orig
ckptdir=ckpt/${dataset}/${ssp}_${gsr}/iter1
logdir=log/${dataset}/${ssp}_${gsr}/iter1

epoch_n=20
subfix=iter1
python cifar_mnist_noisy.py --batch_size ${batch_size} --epoch_num ${epoch_n} --seed $seed \
        --debug 'no' --good_sample_ratio ${gsr} --sub_sampling $ssp --dataset ${dataset}  --model ${model} \
        --learning_rate ${lr} --log_file ${logdir}/${dataset}'_'$model'_gsr_'$gsr'_seed_'$seed'_ssp_'$ssp'_'${subfix}'.log' \
        --loadckpt ${pckptdir}/alg_ckpt_ds_${dataset}_md_${model}_gr_${gsr}_sd_${seed}_ssp_${ssp}_orig.ckpt \
        --empirical_good_sample_ratio ${egsr} --subfix ${subfix} --saveckptpath $ckptdir

mkdir -p ckpt/${dataset}/${ssp}_${gsr}/iter2
mkdir -p log/${dataset}/${ssp}_${gsr}/iter2

pckptdir=ckpt/${dataset}/${ssp}_${gsr}/iter1
ckptdir=ckpt/${dataset}/${ssp}_${gsr}/iter2
logdir=log/${dataset}/${ssp}_${gsr}/iter2


subfix=iter2
python cifar_mnist_noisy.py --batch_size ${batch_size} --epoch_num ${epoch_n} --seed $seed \
        --debug 'no' --good_sample_ratio ${gsr} --sub_sampling $ssp --dataset ${dataset}  --model ${model} \
        --learning_rate ${lr} --log_file ${logdir}/${dataset}'_'$model'_gsr_'$gsr'_seed_'$seed'_ssp_'$ssp'_'${subfix}'.log' \
        --loadckpt ${pckptdir}/alg_ckpt_ds_${dataset}_md_${model}_gr_${gsr}_sd_${seed}_ssp_${ssp}_iter1.ckpt \
        --empirical_good_sample_ratio ${egsr} --subfix ${subfix} --saveckptpath $ckptdir

mkdir -p ckpt/${dataset}/${ssp}_${gsr}/iter3
mkdir -p log/${dataset}/${ssp}_${gsr}/iter3

pckptdir=ckpt/${dataset}/${ssp}_${gsr}/iter2
ckptdir=ckpt/${dataset}/${ssp}_${gsr}/iter3
logdir=log/${dataset}/${ssp}_${gsr}/iter3

subfix=iter3
python cifar_mnist_noisy.py --batch_size ${batch_size} --epoch_num ${epoch_n} --seed $seed \
        --debug 'no' --good_sample_ratio ${gsr} --sub_sampling $ssp --dataset ${dataset}  --model ${model} \
        --learning_rate ${lr} --log_file ${logdir}/${dataset}'_'$model'_gsr_'$gsr'_seed_'$seed'_ssp_'$ssp'_'${subfix}'.log' \
        --loadckpt ${pckptdir}/alg_ckpt_ds_${dataset}_md_${model}_gr_${gsr}_sd_${seed}_ssp_${ssp}_iter2.ckpt \
        --empirical_good_sample_ratio ${egsr} --subfix ${subfix} --saveckptpath $ckptdir

mkdir -p ckpt/${dataset}/${ssp}_${gsr}/iter4
mkdir -p log/${dataset}/${ssp}_${gsr}/iter4

pckptdir=ckpt/${dataset}/${ssp}_${gsr}/iter3
ckptdir=ckpt/${dataset}/${ssp}_${gsr}/iter4
logdir=log/${dataset}/${ssp}_${gsr}/iter4

subfix=iter4
python cifar_mnist_noisy.py --batch_size ${batch_size} --epoch_num ${epoch_n} --seed $seed \
        --debug 'no' --good_sample_ratio ${gsr} --sub_sampling $ssp --dataset ${dataset}  --model ${model} \
        --learning_rate ${lr} --log_file ${logdir}/${dataset}'_'$model'_gsr_'$gsr'_seed_'$seed'_ssp_'$ssp'_'${subfix}'.log' \
        --loadckpt ${pckptdir}/alg_ckpt_ds_${dataset}_md_${model}_gr_${gsr}_sd_${seed}_ssp_${ssp}_iter3.ckpt \
        --empirical_good_sample_ratio ${egsr} --subfix ${subfix} --saveckptpath $ckptdir

mkdir -p ckpt/${dataset}/${ssp}_${gsr}/iter5
mkdir -p log/${dataset}/${ssp}_${gsr}/iter5

pckptdir=ckpt/${dataset}/${ssp}_${gsr}/iter4
ckptdir=ckpt/${dataset}/${ssp}_${gsr}/iter5
logdir=log/${dataset}/${ssp}_${gsr}/iter5

epoch_n=${4}
subfix=iter5
python cifar_mnist_noisy.py --batch_size ${batch_size} --epoch_num ${epoch_n} --seed $seed \
        --debug 'no' --good_sample_ratio ${gsr} --sub_sampling $ssp --dataset ${dataset}  --model ${model} \
        --learning_rate ${lr} --log_file ${logdir}/${dataset}'_'$model'_gsr_'$gsr'_seed_'$seed'_ssp_'$ssp'_'${subfix}'.log' \
        --loadckpt ${pckptdir}/alg_ckpt_ds_${dataset}_md_${model}_gr_${gsr}_sd_${seed}_ssp_${ssp}_iter4.ckpt \
        --empirical_good_sample_ratio ${egsr} --subfix ${subfix} --saveckptpath $ckptdir

mkdir -p ckpt/${dataset}/${ssp}_${gsr}/iter6
mkdir -p log/${dataset}/${ssp}_${gsr}/iter6

pckptdir=ckpt/${dataset}/${ssp}_${gsr}/iter5
ckptdir=ckpt/${dataset}/${ssp}_${gsr}/iter6
logdir=log/${dataset}/${ssp}_${gsr}/iter6

epoch_n=${4}
subfix=iter6
python cifar_mnist_noisy.py --batch_size ${batch_size} --epoch_num ${epoch_n} --seed $seed \
        --debug 'no' --good_sample_ratio ${gsr} --sub_sampling $ssp --dataset ${dataset}  --model ${model} \
        --learning_rate ${lr} --log_file ${logdir}/${dataset}'_'$model'_gsr_'$gsr'_seed_'$seed'_ssp_'$ssp'_'${subfix}'.log' \
        --loadckpt ${pckptdir}/alg_ckpt_ds_${dataset}_md_${model}_gr_${gsr}_sd_${seed}_ssp_${ssp}_iter5.ckpt \
        --empirical_good_sample_ratio ${egsr} --subfix ${subfix} --saveckptpath $ckptdir

mkdir -p ckpt/${dataset}/${ssp}_${gsr}/iter7
mkdir -p log/${dataset}/${ssp}_${gsr}/iter7

pckptdir=ckpt/${dataset}/${ssp}_${gsr}/iter6
ckptdir=ckpt/${dataset}/${ssp}_${gsr}/iter7
logdir=log/${dataset}/${ssp}_${gsr}/iter7

epoch_n=${4}
subfix=iter7
python cifar_mnist_noisy.py --batch_size ${batch_size} --epoch_num ${epoch_n} --seed $seed \
        --debug 'no' --good_sample_ratio ${gsr} --sub_sampling $ssp --dataset ${dataset}  --model ${model} \
        --learning_rate ${lr} --log_file ${logdir}/${dataset}'_'$model'_gsr_'$gsr'_seed_'$seed'_ssp_'$ssp'_'${subfix}'.log' \
        --loadckpt ${pckptdir}/alg_ckpt_ds_${dataset}_md_${model}_gr_${gsr}_sd_${seed}_ssp_${ssp}_iter6.ckpt \
        --empirical_good_sample_ratio ${egsr} --subfix ${subfix} --saveckptpath $ckptdir

mkdir -p ckpt/${dataset}/${ssp}_${gsr}/iter8
mkdir -p log/${dataset}/${ssp}_${gsr}/iter8

pckptdir=ckpt/${dataset}/${ssp}_${gsr}/iter7
ckptdir=ckpt/${dataset}/${ssp}_${gsr}/iter8
logdir=log/${dataset}/${ssp}_${gsr}/iter8

epoch_n=${4}
subfix=iter8
python cifar_mnist_noisy.py --batch_size ${batch_size} --epoch_num ${epoch_n} --seed $seed \
        --debug 'no' --good_sample_ratio ${gsr} --sub_sampling $ssp --dataset ${dataset}  --model ${model} \
        --learning_rate ${lr} --log_file ${logdir}/${dataset}'_'$model'_gsr_'$gsr'_seed_'$seed'_ssp_'$ssp'_'${subfix}'.log' \
        --loadckpt ${pckptdir}/alg_ckpt_ds_${dataset}_md_${model}_gr_${gsr}_sd_${seed}_ssp_${ssp}_iter7.ckpt \
        --empirical_good_sample_ratio ${egsr} --subfix ${subfix} --saveckptpath $ckptdir
