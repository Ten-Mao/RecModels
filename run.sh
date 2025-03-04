# nohup python -u run_sasrec.py --loss_type bpr --device cuda:0 > exception1.log 2>&1 &
# nohup python -u run_hgn.py --loss_type bpr --device cuda:0 > exception2.log 2>&1 &
# nohup python -u run_hgn.py --loss_type ce --device cuda:0 > exception3.log 2>&1 &
# nohup python -u run_gru4rec.py --loss_type bpr --device cuda:0 > exception4.log 2>&1 &
# nohup python -u run_caser.py --loss_type bpr --device cuda:1 > exception5.log 2>&1 &
nohup python -u run_caser.py --loss_type bpr --device cuda:0 > exception.log 2>&1 &
# nohup python -u run_bpr.py --loss_type bpr --device cuda:1 > exception7.log 2>&1 &