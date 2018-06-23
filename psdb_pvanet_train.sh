basepath=$(cd `dirname $0`; pwd)
./train.py \
    --train-path ${basepath}/data/psdb/train.rec \
    --num-class 4 \
    --prefix ${basepath}/output/psdbPvanet/ssd \
    --val-path ${basepath}/data/psdb/val.rec \
	--label-width 1200 \
	--network pvanet \
	--freeze ''\
	--data-shape 300 \
    --class-names 'pedestrian, head, head-shouler, upper-body' 

