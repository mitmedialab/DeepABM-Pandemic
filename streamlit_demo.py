import streamlit as st
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

from main_run import runner

COLOR_VALS = ['red', 'green', 'blue', 'orange', 'purple', 'black', 'gray', 'yellow', 'cyan', 'hotpink', 'lime', 'teal']

st.title('COVID-19 Vaccine ABM')


def log_all_data(d, stage, n=3000):
    fnames = glob.glob(d + '/*.csv')
    if len(fnames) > 0:
        df = pd.read_csv(fnames[0])
        res = df[stage].values
        data = np.zeros((res.shape[0], len(fnames)))
        data[:, 0] = res

        ctr = 0
        for f in fnames[1:]:
            ctr += 1
            df = pd.read_csv(f)
            tmp_val = df[stage].values
            res += tmp_val
            data[:, ctr] = tmp_val

        res = res / len(fnames)
        return res, len(fnames), np.array(data)

    return None, None


def visualize(results_root, params=None, plotmedian=True):
    global COLOR_VALS

    subplot_rows = 2
    num_agents = 100000 # params['num_agents']
    stages = ['INFECTED', 'DEATH', 'ACTIVE', 'HOSPITALIZED']

    file_list = sorted(glob.glob(results_root + '/*'))

    print(" --------------- File Lists: ", file_list, " ---------------")

    select_index = range(len(file_list))
    num_agents_list = [num_agents] * len(select_index)
    labels = [file_list[i].split('_')[-1] for i in select_index]
    dt = [file_list[i] for i in select_index]

    fig, axs = plt.subplots(subplot_rows, subplot_rows, figsize=(10,10))
    x_ix, y_ix = 0, 0

    ctr = 0
    legend_lines = []
    names = []
    for stage in stages:
        print("Doing stage: ", stage)

        x_ix, y_ix = int(ctr//subplot_rows), int(ctr%subplot_rows)

        print(x_ix, y_ix)

        ctr += 1

        for i, d in enumerate(dt):
            n = num_agents_list[i]
            print(n)
            res, num_runs, all_data = log_all_data(d, stage, n)

            if res is not None:
                if plotmedian:
                    title = stage #'Median and 25-75 percentiles'
                    mn_vals = np.median(all_data, axis=1)
                    std_vals = np.std(all_data, axis=1)
                    under_line = np.percentile(all_data, 25, axis=1)
                    over_line = np.percentile(all_data, 75, axis=1)
                else:
                    title = stage  #'Mean and 1 standard deviation'
                    mn_vals = np.mean(all_data, axis=1)
                    std_vals = np.std(all_data, axis=1)
                    under_line = mn_vals - 1 * std_vals
                    over_line = mn_vals + 1 * std_vals

                axs[x_ix, y_ix].plot(mn_vals, linewidth=2.5, color=COLOR_VALS[i], label='{}({} runs)'.format(labels[i], num_runs))
                axs[x_ix, y_ix].plot(under_line, linewidth=0.5, color=COLOR_VALS[i])
                line, = axs[x_ix, y_ix].plot(over_line, linewidth=0.5, color=COLOR_VALS[i])
                axs[x_ix, y_ix].set_title(title)
                axs[x_ix, y_ix].fill_between(range(mn_vals.shape[0]), under_line, over_line, color=COLOR_VALS[i], alpha=.1)  # std curves.)

                legend_lines.append(line)
                if labels[i] == 'Baseline':
                    names.append(labels[i])
                else:
                    names.append('+' + labels[i])

        axs[x_ix, y_ix].set_xlabel('days', fontsize=10)

        if stage == 'Vaccinated':
            axs[x_ix, y_ix].set_ylabel('Atleast 1 Vaccine Dose', fontsize=16)
        else:
            axs[x_ix, y_ix].set_ylabel('# subjects', fontsize=16)

    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), #ncol=3, fancybox=True, shadow=True)

    fig.legend(legend_lines[:len(select_index)], names[:len(select_index)], prop={'size': 18}, loc='upper center', fancybox=True, shadow=True, bbox_to_anchor=(1.0, 0.5))

    fig.tight_layout()

    st.pyplot(fig)

    return


def run(flags_dict):

    # results_root, params = runner(flags_dict)

    pwd = os.getcwd()
    print("Pwd: ", pwd)
    results_root = os.path.join(pwd, 'Data/Results/')
    print("Results root: ", results_root)

    visualize(results_root)

    # st.write('on button click')
    return


def run_visual(flags_dict):
    results_root = os.getcwd()

    checked_stats = sorted([k for k in flags_dict.keys() if flags_dict[k]])



    folder_name = ''.join(checked_stats)

    folder_path = os.path.join(results_root, folder_name)

    visualize(folder_path)


stocks = ["Quarantine", "DEN", "Fast Test", "Vaccine 1 Dose", "Vaccine 2 Dose"]

st.sidebar.markdown('# COMPARE INTERVENTIONS\n')
st.sidebar.markdown('\n')

check_boxes = [st.sidebar.checkbox(stock, key=stock) for stock in stocks]

flags_dict = dict()
for stock, checked in zip(stocks, check_boxes):
    stock_name = ''.join(stock.split(' '))
    flags_dict[stock_name] = checked

print(flags_dict)

if st.button('Run Experiment'):
    run_visual(flags_dict)

# st.write([stock for stock, checked in zip(stocks, check_boxes) if checked])
