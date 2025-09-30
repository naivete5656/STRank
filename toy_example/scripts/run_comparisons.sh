
for func_param in 0 1 2 3; do
    python ./toy_example/main.py --data_type multi --n_sample 50000 --r 100 --scale 1 --bias 0 --func_param $func_param --sampling uniform --save_dir ./outputs/synthetic_dat --epochs 200 --lr 1e-3; 
done