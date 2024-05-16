cd code &&
python main.py --dataset="ml-20m" --model="DAN" --reg_p=0.05 --alpha=0.8 --beta=0.4 --xi=0.1 --diag_const=False --diag_relax=True &&
python main.py --dataset="netflix" --model="DAN" --reg_p=0.01 --alpha=0.8 --beta=0.7 --xi=0.1 --diag_const=False --diag_relax=True &&
python main.py --dataset="msd" --model="DAN" --reg_p=0.05 --alpha=0.9 --beta=0.4 --xi=0.1 --diag_const=False --diag_relax=True &&
python main.py --dataset="gowalla" --model="DAN" --reg_p=5 --alpha=0.8 --beta=0.2 --xi=0.9 --diag_const=False --diag_relax=True &&
python main.py --dataset="yelp2018" --model="DAN" --reg_p=5 --alpha=0.6 --beta=0.3 --xi=0.1 --diag_const=False --diag_relax=True &&
python main.py --dataset="abook" --model="DAN" --reg_p=1 --alpha=0.7 --beta=0.4 --xi=0.7 --diag_const=False --diag_relax=True 