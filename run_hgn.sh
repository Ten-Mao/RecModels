nohup python -u run_hgn.py --device cuda:0 --dataset Beauty2014 --d_model 128 --pool_type avg --lr 1e-3  --wd 1e-4 > log/Beauty2014/hgn-d_model_128-pool_type_avg-lr_1e-3-wd_1e-4.log 2>&1 &