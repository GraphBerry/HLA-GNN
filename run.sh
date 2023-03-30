运行消融实验，标记和标记重使用
nohup python main_homo.py --use_labels --m 0.3 -d cornell>logs/lp_reuse/cornell.log &

nohup python main_homo.py --use_labels --m 0.3 -d cornell>logs/S_lp_reuse/cornell.log &

python main_homo.py --use_labels --m 0.4 -d cora

python main_homo.py --use_labels --label_reuse --m 0.4 -d cora