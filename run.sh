# nohup python -u run_sasrec.py --loss_type ce --lr 5e-4 --weight_decay 5e-1 --device cuda:0 > /dev/null 2>&1 &
# nohup python -u run_gru4rec.py --loss_type ce --lr 1e-3 --weight_decay 5e-1 --device cuda:0 > /dev/null 2>&1 &
nohup python -u run_caser.py --loss_type ce --lr 5e-4 --weight_decay 1 --device cuda:1 > /dev/null 2>&1 &
