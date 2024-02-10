
# nohup python -u main.py --params Params/params_no_inter.yaml --seed 1234 --num_runs 10 --results_file_postfix NI> logs/NI.txt &
# nohup python -u main.py --params Params/params_sq.yaml --seed 1234 --num_runs 10 --results_file_postfix SQ> logs/SQ.txt &
# nohup python -u main.py --params Params/params_vacc.yaml --seed 1234 --num_runs 10 --results_file_postfix VACC> logs/VACC.txt &
# nohup python -u main.py --params Params/params_ct.yaml --seed 1234 --num_runs 10 --results_file_postfix CT> logs/CT.txt &

##########Full dynamics ##############
# nohup python -u main.py --params Params/params_no_inter.yaml --seed 1234 --num_runs 5s --log_full_dynamics --results_file_postfix NI_full > logs/NI_full.txt &
# nohup python -u main.py --params Params/params_sq.yaml --seed 1234 --num_runs 5 --log_full_dynamics --results_file_postfix SQ_full > logs/SQ_full.txt &
# nohup python -u main.py --params Params/params_vacc.yaml --seed 1234 --num_runs 5 --log_full_dynamics --results_file_postfix VACC_full > logs/VACC_full.txt &
# nohup python -u main.py --params Params/params_ct.yaml --seed 1234 --num_runs 5 --log_full_dynamics --results_file_postfix CT_full > logs/CT_full.txt &
# nohup python -u main.py --params Params/params_vacc_ct.yaml --seed 1234 --num_runs 5 --log_full_dynamics --results_file_postfix VACC_CT_full > logs/VACC_CT_full.txt &

# ##########Vaccination###############
# nohup python -u main.py --params Params/params_vacc.yaml --seed 1234 --num_runs 10 --results_file_postfix VACC_30  --vaccine_start_date 30 > logs/VACC_only_30.txt &
# nohup python -u main.py --params Params/params_vacc.yaml --seed 1234 --num_runs 10 --results_file_postfix VACC_60  --vaccine_start_date 60 > logs/VACC_only_60.txt &

# nohup python -u main.py --params Params/params_vacc.yaml --seed 1234 --num_runs 10 --results_file_postfix VACC_30_eff_7  --vaccine_start_date 30 --vaccine_first_dose_effectivness 0.7> logs/VACC_30_eff_7.txt &
# nohup python -u main.py --params Params/params_vacc.yaml --seed 1234 --num_runs 10 --results_file_postfix VACC_30_eff_5 --vaccine_start_date 30 --vaccine_first_dose_effectivness 0.5 > logs/VACC_30_eff_5.txt &
# nohup python -u main.py --params Params/params_vacc.yaml --seed 1234 --num_runs 10 --results_file_postfix VACC_30_eff_3 --vaccine_start_date 30 --vaccine_first_dose_effectivness 0.3 > logs/VACC_30_eff_3.txt &

# nohup python -u main.py --params Params/params_vacc_sq_test.yaml --seed 1234 --num_runs 10 --results_file_postfix VACC_sq_test_10  --vaccine_start_date 10 > logs/vacc_10.txt &
# nohup python -u main.py --params Params/params_vacc_sq_test.yaml --seed 1234 --num_runs 10 --results_file_postfix VACC_sq_test_30  --vaccine_start_date 30 > logs/vacc_30.txt &
# nohup python -u main.py --params Params/params_vacc_sq_test.yaml --seed 1234 --num_runs 10 --results_file_postfix VACC_sq_test_60  --vaccine_start_date 60 > logs/vacc_60.txt &

# nohup python -u main.py --params Params/params_vacc_sq_test.yaml --seed 1234 --num_runs 10 --results_file_postfix VACC_sq_test_30_eff_7  --vaccine_start_date 30 --vaccine_first_dose_effectivness 0.7 > logs/vacc_30_eff_7.txt &
# nohup python -u main.py --params Params/params_vacc_sq_test.yaml --seed 1234 --num_runs 10 --results_file_postfix VACC_sq_test_30_eff_5  --vaccine_start_date 30 --vaccine_first_dose_effectivness 0.5 > logs/vacc_30_eff_5.txt &
# nohup python -u main.py --params Params/params_vacc_sq_test.yaml --seed 1234 --num_runs 10 --results_file_postfix VACC_sq_test_30_eff_3  --vaccine_start_date 30 --vaccine_first_dose_effectivness 0.3 > logs/vacc_30_eff_3.txt &


# nohup python -u main.py --params Params/params_vacc_ct.yaml --seed 1234 --num_runs 10 --results_file_postfix VACC_CT_30  --vaccine_start_date 30 > logs/VACC_CT_30.txt &
# nohup python -u main.py --params Params/params_vacc_ct.yaml --seed 1234 --num_runs 10 --results_file_postfix VACC_CT_60  --vaccine_start_date 60 > logs/VACC_CT_60.txt &

# nohup python -u main.py --params Params/params_vacc.yaml --seed 1234 --num_runs 10 --vaccine_daily_production 124 --results_file_postfix VACC_daily_124 > logs/VACC_daily_124.txt &
# nohup python -u main.py --params Params/params_vacc_ct.yaml --seed 1234 --num_runs 10 --results_file_postfix VACC_CT_30_daily_100 --vaccine_start_date 30 --vaccine_daily_production 100> logs/VACC_CT_30_daily_100.txt &

# # ##########Testing###############
# nohup python -u main.py --params Params/params_sq.yaml --seed 1234 --num_runs 10 --results_file_postfix SQ_test_10 --rtpcr_test_start_date 10 > logs/sq_10.txt &
# nohup python -u main.py --params Params/params_sq.yaml --seed 1234 --num_runs 10 --results_file_postfix SQ_test_30 --rtpcr_test_start_date 30 > logs/sq_30.txt &
# nohup python -u main.py --params Params/params_sq.yaml --seed 1234 --num_runs 10 --results_file_postfix SQ_test_60 --rtpcr_test_start_date 60 > logs/sq_60.txt &

# nohup python -u main.py --params Params/params_ct.yaml --seed 1234 --num_runs 10 --rtpcr_test_start_date 30 --results_file_postfix CT_30 > logs/CT_30.txt &
# nohup python -u main.py --params Params/params_ct.yaml --seed 1234 --num_runs 10 --rtpcr_test_start_date 60 --results_file_postfix CT_60 > logs/CT_60.txt &


#################Sensitivity Analysis################
# # 1. app_adoption_rate
nohup python -u main.py --params Params/params_ct.yaml --seed 1234 --num_runs 5 --app_adoption_rate 0.2 --results_file_postfix CT_app_20> logs/CT_app_20.txt &
nohup python -u main.py --params Params/params_ct.yaml --seed 1234 --num_runs 5 --app_adoption_rate 0.6 --results_file_postfix CT_app_60> logs/CT_app_60.txt &
nohup python -u main.py --params Params/params_ct.yaml --seed 1234 --num_runs 5 --app_adoption_rate 0.8 --results_file_postfix CT_app_80> logs/CT_app_80.txt &

# 2. en_quarantine_enter_prob
# nohup python -u main.py --params Params/params_ct.yaml --seed 1234 --num_runs 5 --en_quarantine_enter_prob 0.4 --results_file_postfix CT_en_sq_40> logs/CT_en_sq_40.txt &
# nohup python -u main.py --params Params/params_ct.yaml --seed 1234 --num_runs 5 --en_quarantine_enter_prob 0.6 --results_file_postfix CT_en_sq_60> logs/CT_en_sq_60.txt &
# # nohup python -u main.py --params Params/params_ct.yaml --seed 1234 --num_runs 5 --en_quarantine_enter_prob 0.8 --results_file_postfix CT_en_sq_80> logs/CT_en_sq_80.txt &
# nohup python -u main.py --params Params/params_ct.yaml --seed 1234 --num_runs 5 --en_quarantine_enter_prob 1.0 --results_file_postfix CT_en_sq_100> logs/CT_en_sq_100.txt &

# 3. vaccine_first_dose_effectivness
# nohup python -u main.py --params Params/params_vacc.yaml --seed 1234 --num_runs 5 --vaccine_first_dose_effectivness 0.9 --results_file_postfix VACC_eff1_90> logs/VACC_eff1_90.txt &
# nohup python -u main.py --params Params/params_vacc.yaml --seed 1234 --num_runs 5 --vaccine_first_dose_effectivness 0.7 --results_file_postfix VACC_eff1_70> logs/VACC_eff1_70.txt &
# nohup python -u main.py --params Params/params_vacc.yaml --seed 1234 --num_runs 5 --vaccine_first_dose_effectivness 0.5 --results_file_postfix VACC_eff1_50> logs/VACC_eff1_50.txt &
# nohup python -u main.py --params Params/params_vacc.yaml --seed 1234 --num_runs 5 --vaccine_first_dose_effectivness 0.3 --results_file_postfix VACC_eff1_30> logs/VACC_eff1_30.txt &





