basepath=$(cd `dirname $0`; pwd)
./evaluate_mobi.py \
    --rec-path ${basepath}/data/psdb/val.rec \
	--network mobilenetLess \
    --num-class 4 \
	--data-shape 300 \
	--cpu \
	--epoch 240 \
	--batch-size 1 \
    --prefix ${basepath}/output/psdbMobileNetLess/ssd \
    --class-names 'pedestrian, head, head-shouler, upper-body'  &> log_eval_psdb_mobiLess &


	#--gpus 0 \
