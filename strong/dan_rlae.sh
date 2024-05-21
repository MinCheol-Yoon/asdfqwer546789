cd code &&
# python main.py --dataset="ml-20m" --model="DAN" --reg_p=0.05 --alpha=0.8 --beta=0.5 --xi=0.3 --relax=True &&
# python main.py --dataset="netflix" --model="DAN" --reg_p=0.01 --alpha=0.8 --beta=0.7 --xi=0.1 --diag_const=False --diag_relax=True &&
# python main.py --dataset="msd" --model="DAN" --reg_p=0.05 --alpha=1 --beta=0.4 --xi=0.1 --diag_const=False --diag_relax=True &&
python main.py --dataset="gowalla" --model="DAN_diag" --reg_p=4 --alpha=0.8 --beta=0.2 --xi=0.9 --relax=True &&
python main.py --dataset="yelp2018" --model="DAN_diag" --reg_p=8 --alpha=0.6 --beta=0.3 --xi=0.1 --relax=True &&
python main.py --dataset="abook" --model="DAN_diag" --reg_p=3 --alpha=0.8 --beta=0.0 --xi=0.9 --relax=True
