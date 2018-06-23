basepath=$(cd `dirname $0`; pwd)
./evaluate.py \
    --rec-path ${basepath}/data/psdb/val.rec \
    --num-class 4 \
	--data-shape 300 \
	--gpus 0 \
	--epoch 240 \
	--batch-size 1 \
	--voc07 False \
    --prefix ${basepath}/output/psdb/ssd \
    --class-names 'pedestrian, head, head-shouler, upper-body'  #&> log_eval_psdb &

	#--cpu \
