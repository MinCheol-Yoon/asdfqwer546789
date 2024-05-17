cd code &&
python main.py --dataset="gowalla" --model="DAN" --reg_p=3 --alpha=1 --beta=0.6 --xi=0.9 --diag_const=False --diag_relax=True &&
python main.py --dataset="yelp2018" --model="DAN" --reg_p=4 --alpha=0.6 --beta=0.5 --xi=0.2 --diag_const=False --diag_relax=True &&
python main.py --dataset="abook" --model="DAN" --reg_p=1 --alpha=0.5 --beta=0.3 --xi=0.1 --diag_const=False --diag_relax=True 