DOSAGE_VALUE=07

nohup python -u main_steril.py --params Data/generated_params-expvac-2.yaml --seed 1234 --num_runs 10 --results_file_postfix NoDenYesQYesVac1FS20EFFSTERIL${DOSAGE_VALUE}STERILFIG  > logs/tmp_NoDenYesQYesVac1FS20_${DOSAGE_VALUE}.txt &
nohup python -u main_steril.py --params Data/generated_params-expvac-3.yaml --seed 1234 --num_runs 10 --results_file_postfix NoDenYesQYesVac2FS20EFFSTERIL${DOSAGE_VALUE}STERILFIG  > logs/tmp_NoDenYesQYesVac2FS20_${DOSAGE_VALUE}.txt &
