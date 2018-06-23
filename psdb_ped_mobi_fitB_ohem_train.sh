basepath=$(cd `dirname $0`; pwd)
./train_ohem.py \
    --train-path ${basepath}/data/psdb_ped/train.rec \
    --num-class 1 \
    --prefix ${basepath}/output/psdbPedMobileNet_fitB/ssd \
    --val-path ${basepath}/data/psdb_ped/val.rec \
	--label-width 1200 \
	--network mobilenet_fitB \
	--pretrained ${basepath}/model/mobilenet-ssd-512 \
	--finetune 1 \
	--data-shape 300 \
	--gpu 0 \
    --class-names 'pedestrian' 

#	--lr 0.00001 \
#	--batch-size 1 \
#	--frequent 1 \
