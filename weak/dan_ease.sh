cd code &&
python main.py --dataset="gowalla" --model="DAN_diag" --reg_p=2 --alpha=0.9 --beta=0.8 --relax=False&&
python main.py --dataset="yelp2018" --model="DAN_diag" --reg_p=4 --alpha=0.6 --beta=0.5 --relax=False&&
python main.py --dataset="abook" --model="DAN_diag" --reg_p=1 --alpha=0.5 --beta=0.3 --relax=False