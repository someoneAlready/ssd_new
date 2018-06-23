basepath=$(cd `dirname $0`; pwd)
python tools/prepare_dataset.py --dataset psdb_ped --set train --target ./data/psdb_ped/train.lst --root ${basepath}/../data/psdb_ped --shuffle True
python tools/prepare_dataset.py --dataset psdb_ped --set test --target ./data/psdb_ped/val.lst  --root ${basepath}/../data/psdb_ped

