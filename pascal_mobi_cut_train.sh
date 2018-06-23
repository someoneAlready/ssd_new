basepath=$(cd `dirname $0`; pwd)
./train_cut.py \
    --prefix ${basepath}/output/pascalMobileNet_cut/ssd \
	--network mobilenet_cut \
    --pretrained ${basepath}/model/mobilenet-ssd-512 \
    --finetune 1 \
	--data-shape 512 \
	--batch-size 32

