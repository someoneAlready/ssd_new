basepath=$(cd `dirname $0`; pwd)
./train.py \
    --train-path ${basepath}/data/psdb_ped/train.rec \
    --num-class 1 \
    --prefix ${basepath}/output/psdbPedMobileNet_fit/ssd \
    --val-path ${basepath}/data/psdb_ped/val.rec \
	--label-width 1200 \
	--network mobilenet_fit \
	--pretrained ${basepath}/model/mobilenet-ssd-512 \
	--finetune 1 \
	--data-shape 300 \
    --class-names 'pedestrian' 

