DOSAGE_VALUE=08

nohup python -u main_agedep_2dose.py --params Data/generated_params-expvac-2.yaml --seed 1234 --num_runs 10 --results_file_postfix Vac1FS20EFFAgeBase${DOSAGE_VALUE}333 > logs/tmp_AgeBaseNoDenYesQYesVac1FS20_${DOSAGE_VALUE}.txt &
nohup python -u main_agedep_2dose.py --params Data/generated_params-expvac-3.yaml --seed 1234 --num_runs 10 --results_file_postfix NoDenYesQYesVac2FS20EFAgeBaseF${DOSAGE_VALUE}333  > logs/tmp_AgeBaseNoDenYesQYesVac2FS20_${DOSAGE_VALUE}.txt &


