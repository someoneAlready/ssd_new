basepath=$(cd `dirname $0`; pwd)
./evaluate.py \
    --rec-path ${basepath}/data/psdb_ped/val.rec \
	--network mobilenet_fit \
    --num-class 1 \
	--data-shape 300 \
	--gpus 0 \
	--epoch 240 \
	--batch-size 100 \
    --prefix ${basepath}/output/psdbPedMobileNet_fit/ssd \
    --class-names 'pedestrian' 


#	--cpu \
