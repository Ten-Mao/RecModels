# nohup python -u run_tiger.py  --device cuda:0 --dataset Beauty2014 --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position last --t54rec_lr 5e-4 --t54rec_train_batch_size 256 --kmeans_init_iter 10 > log/Beauty2014/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_last-t54rec_lr_5e-4-t54rec_train_batch_size_256.log 2>&1 &
# nohup python -u run_tiger.py  --device cuda:0 --dataset Beauty2014 --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position last --t54rec_lr 1e-3 --t54rec_train_batch_size 256 --kmeans_init_iter 10 > log/Beauty2014/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_last-t54rec_lr_1e-3-t54rec_train_batch_size_256.log 2>&1 &
# nohup python -u run_tiger.py  --device cuda:0 --dataset Beauty2014 --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position last --t54rec_lr 1e-3 --t54rec_train_batch_size 256 --kmeans_init_iter 10 > log/Beauty2014/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_last-t54rec_lr_1e-3-t54rec_train_batch_size_256.log 2>&1 &
# nohup python -u run_tiger.py  --device cuda:0 --dataset Beauty2014 --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position best --t54rec_lr 5e-4 --t54rec_train_batch_size 256 --kmeans_init_iter 10 > log/Beauty2014/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_best-t54rec_lr_5e-4-t54rec_train_batch_size_256.log 2>&1 &
# nohup python -u run_tiger.py  --device cuda:0 --dataset Beauty2014 --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position best --t54rec_lr 1e-3 --t54rec_train_batch_size 256 --kmeans_init_iter 10 > log/Beauty2014/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_best-t54rec_lr_1e-3-t54rec_train_batch_size_256.log 2>&1 &

# nohup python -u run_tiger.py  --device cuda:1 --dataset Beauty --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position last --t54rec_lr 5e-4 --t54rec_train_batch_size 256 --kmeans_init_iter 10 --sinkhorn_open > log/Beauty/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_last-t54rec_lr_5e-4-t54rec_train_batch_size_256.log 2>&1 &
# nohup python -u run_tiger.py  --device cuda:2 --dataset Arts --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position last --t54rec_lr 5e-4 --t54rec_train_batch_size 256 --kmeans_init_iter 10 --sinkhorn_open > log/Arts/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_last-t54rec_lr_5e-4-t54rec_train_batch_size_256.log 2>&1 &
# nohup python -u run_tiger.py  --device cuda:3 --dataset Instruments --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position last --t54rec_lr 5e-4 --t54rec_train_batch_size 256 --kmeans_init_iter 10 --sinkhorn_open > log/Instruments/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_last-t54rec_lr_5e-4-t54rec_train_batch_size_256.log 2>&1 &


nohup python -u run_tiger.py  --device cuda:0 --dataset Beauty --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position last --t54rec_lr 1e-3 --t54rec_train_batch_size 256 --kmeans_init_iter 10 --sinkhorn_open > log/Beauty/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_last-t54rec_lr_1e-3-t54rec_train_batch_size_256.log 2>&1 &
nohup python -u run_tiger.py  --device cuda:1 --dataset Beauty --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position last --t54rec_lr 1e-3 --t54rec_train_batch_size 256 --kmeans_init_iter 10 --sinkhorn_open > log/Beauty/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_best-t54rec_lr_1e-3-t54rec_train_batch_size_256.log 2>&1 &
nohup python -u run_tiger.py  --device cuda:2 --dataset Beauty --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position best --t54rec_lr 5e-4 --t54rec_train_batch_size 256 --kmeans_init_iter 10 --sinkhorn_open > log/Beauty/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_best-t54rec_lr_5e-4-t54rec_train_batch_size_256.log 2>&1 &

nohup python -u run_tiger.py  --device cuda:3 --dataset Arts --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position last --t54rec_lr 1e-3 --t54rec_train_batch_size 256 --kmeans_init_iter 10 --sinkhorn_open > log/Arts/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_last-t54rec_lr_1e-3-t54rec_train_batch_size_256.log 2>&1 &
nohup python -u run_tiger.py  --device cuda:4 --dataset Arts --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position best --t54rec_lr 1e-3 --t54rec_train_batch_size 256 --kmeans_init_iter 10 --sinkhorn_open > log/Arts/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_best-t54rec_lr_1e-3-t54rec_train_batch_size_256.log 2>&1 &
nohup python -u run_tiger.py  --device cuda:5 --dataset Arts --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position best --t54rec_lr 5e-4 --t54rec_train_batch_size 256 --kmeans_init_iter 10 --sinkhorn_open > log/Arts/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_best-t54rec_lr_5e-4-t54rec_train_batch_size_256.log 2>&1 &

nohup python -u run_tiger.py  --device cuda:6 --dataset Instruments --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position last --t54rec_lr 1e-3 --t54rec_train_batch_size 256 --kmeans_init_iter 10 --sinkhorn_open > log/Instruments/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_last-t54rec_lr_1e-3-t54rec_train_batch_size_256.log 2>&1 &
nohup python -u run_tiger.py  --device cuda:7 --dataset Instruments --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position best --t54rec_lr 1e-3 --t54rec_train_batch_size 256 --kmeans_init_iter 10 --sinkhorn_open > log/Instruments/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_best-t54rec_lr_1e-3-t54rec_train_batch_size_256.log 2>&1 &
nohup python -u run_tiger.py  --device cuda:0 --dataset Instruments --rqvae_lr 1e-3 --rqvae_wd 1e-4 --rqvae_select_position best --t54rec_lr 5e-4 --t54rec_train_batch_size 256 --kmeans_init_iter 10 --sinkhorn_open > log/Instruments/tiger-rqvae_lr_1e-3-rqvae_wd_1e-4-kmeans_init_iter_10-rqvae_select_position_best-t54rec_lr_5e-4-t54rec_train_batch_size_256.log 2>&1 &











