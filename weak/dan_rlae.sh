cd code &&
python main.py --dataset="gowalla" --model="DAN_diag" --reg_p=3 --alpha=1 --beta=0.6 --xi=0.9 --relax=True &&
python main.py --dataset="yelp2018" --model="DAN_diag" --reg_p=4 --alpha=0.6 --beta=0.5 --xi=0.2 --relax=True &&
python main.py --dataset="abook" --model="DAN_diag" --reg_p=1 --alpha=0.5 --beta=0.3 --xi=0.1 --relax=True 