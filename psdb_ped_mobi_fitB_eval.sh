basepath=$(cd `dirname $0`; pwd)
./evaluate.py \
    --rec-path ${basepath}/data/psdb_ped/val.rec \
	--network mobilenet_fitB \
    --num-class 1 \
	--data-shape 300 \
	--gpus 0 \
	--epoch 240 \
	--batch-size 1 \
    --prefix ${basepath}/output/psdbPedMobileNet_fitB/ssd \
    --class-names 'pedestrian' 


#	--cpu \
