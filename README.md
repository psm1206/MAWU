# Toward a Better Understanding of Loss Functions for Collaborative Filtering (CIKM'23)

For more information about our paper, please follow [Toward a Better Understanding of Loss Functions for Collaborative Filtering](https://arxiv.org/abs/2308.06091).
This code is implemented on [RecBole](https://github.com/RUCAIBox/RecBole).

## How to run

### Set conda environment
```
conda env create -f mawu.yaml
conda activate mawu
```

### Run codes for DirectAU
```
python run_recbole.py --model=DirectAU --dataset=beauty --encoder=MF --weight_decay=1e-4 --gamma=0.4 &&
python run_recbole.py --model=DirectAU --dataset=beauty --encoder=LightGCN --weight_decay=1e-4 --gamma=0.4 &&
python run_recbole.py --model=DirectAU --dataset=gowalla --encoder=MF --weight_decay=1e-6 --gamma=2 &&
python run_recbole.py --model=DirectAU --dataset=gowalla --encoder=LightGCN --weight_decay=1e-6 --gamma=2 &&
python run_recbole.py --model=DirectAU --dataset=yelp --encoder=MF --weight_decay=1e-6 --gamma=2 &&
python run_recbole.py --model=DirectAU --dataset=yelp --encoder=LightGCN --weight_decay=1e-6 --gamma=2
```

### Run codes for MAWU
```
python run_recbole.py --model=MAWU --dataset=beauty --encoder=MF --weight_decay=1e-4 --gamma1=1 --gamma2=0.1 &&
python run_recbole.py --model=MAWU --dataset=beauty --encoder=LightGCN --weight_decay=1e-4 --gamma1=0.9 --gamma2=0.2 &&
python run_recbole.py --model=MAWU --dataset=gowalla --encoder=MF --weight_decay=1e-6 --gamma1=2.6 --gamma2=1.4 &&
python run_recbole.py --model=MAWU --dataset=gowalla --encoder=LightGCN --weight_decay=1e-6 --gamma1=2.4 --gamma2=1.6 &&
python run_recbole.py --model=MAWU --dataset=yelp --encoder=MF --weight_decay=1e-6 --gamma1=0.8 --gamma2=0.6 &&
python run_recbole.py --model=MAWU --dataset=yelp --encoder=LightGCN --weight_decay=1e-6 --gamma1=1.2 --gamma2=0.6
```

## Citation
If you find our work helpful, please cite our paper.
```
@inproceedings{park2023mawu,
  title={Toward a Better Understanding of Loss Functions for Collaborative Filtering},
  author={Seongmin Park and
          Mincheol Yoon and
          Jae-woong Lee and
          Hogun Park and
          Jongwuk Lee},
  booktitle={The 32nd ACM International Conference on Information and Knowledge Management (CIKM)},
  year={2023}
}
```
