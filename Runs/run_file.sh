DOSAGE_VALUE=08

nohup python -u main.py --params Data/generated_params-expvac-2.yaml --seed 1234 --num_runs 10 --results_file_postfix VAC_100_1FEXPT08  > logs/tmp_fig1_1f_expt_100vac.txt &
nohup python -u main.py --params Data/generated_params-expvac-3.yaml --seed 1234 --num_runs 10 --results_file_postfix  VAC_100_2FEXPT08 > logs/tmp_fig1_2f_expt_100vac.txt &