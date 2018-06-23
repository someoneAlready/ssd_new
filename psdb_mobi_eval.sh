basepath=$(cd `dirname $0`; pwd)
./evaluate.py \
    --rec-path ${basepath}/data/psdb/val.rec \
	--network mobilenet \
    --num-class 4 \
	--data-shape 300 \
	--gpus 0 \
	--epoch 240 \
	--batch-size 100 \
    --prefix ${basepath}/output/psdbMobileNet/ssd \
    --class-names 'pedestrian, head, head-shouler, upper-body' #  &> log_eval_psdb_mobi &


#	--cpu \
