cd code &&
python main.py --dataset="gowalla" --model="DAN" --reg_p=2 --alpha=0.9 --beta=0.8 --diag_const=False --diag_relax=False &&
python main.py --dataset="yelp2018" --model="DAN" --reg_p=4 --alpha=0.6 --beta=0.5 --diag_const=False --diag_relax=False &&
python main.py --dataset="abook" --model="DAN" --reg_p=1 --alpha=0.5 --beta=0.3 --diag_const=False --diag_relax=False 