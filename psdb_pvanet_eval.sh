basepath=$(cd `dirname $0`; pwd)
./evaluate.py \
    --rec-path ${basepath}/data/psdb/val.rec \
	--network pvanet \
    --num-class 4 \
	--data-shape 300 \
	--gpus 0 \
	--epoch 240 \
	--batch-size 1 \
    --prefix ${basepath}/output/psdbPvanet/ssd \
    --class-names 'pedestrian, head, head-shouler, upper-body'   &> log_eval_psdb_pvanet &


#	--cpu \
