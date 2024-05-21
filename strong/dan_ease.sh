cd code &&
# python main.py --dataset="ml-20m" --model="DAN" --reg_p=0.05 --alpha=0.8 --beta=0.4 --diag_const=True --diag_relax=False &&
# python main.py --dataset="netflix" --model="DAN" --reg_p=0.01 --alpha=0.8 --beta=0.7 --diag_const=True --diag_relax=False &&
# python main.py --dataset="msd" --model="DAN" --reg_p=0.05 --alpha=0.9 --beta=0.5 --diag_const=True --diag_relax=False &&
python main.py --dataset="gowalla" --model="DAN_diag" --reg_p=5 --alpha=0.8 --beta=0.1 --relax=False &&
python main.py --dataset="yelp2018" --model="DAN_diag" --reg_p=5 --alpha=0.6 --beta=0.4 --relax=False &&
python main.py --dataset="abook" --model="DAN_diag" --reg_p=3 --alpha=0.8 --beta=0.0 --relax=False
