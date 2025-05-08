import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))

name_mapping = {
    'ud': 'CrossUD',
    'subud': 'CrossUDSub',
    'wave': 'CrossWave',
    'ZBH1': 'ZBH1',
    '1F1B': '1F1B',
    'ZBV': 'ZBV',
    'ZBH1_P': 'ZBH1_O',
    '1F1B_P': '1F1B_O',
    'ZBV_P': 'ZBV_O',
}

schedule_styles = {
    '1F1B': {
        'marker': '*',
        'color': '#1976d2', 
        'linewidth': 2,
        'markersize': 6,
        'linestyle': ':',
        'alpha': 0.8,
        'zorder': 12
    },
    'ZBH1': {
        'marker': 'P',
        'color': '#42a5f5',
        'linewidth': 2,
        'markersize': 6,
        'linestyle': ':',
        'alpha': 0.8,
        'zorder': 12
    },
    'ZBV': {
        'marker': '^',
        'color': '#ff7f7f',
        'linewidth': 2,
        'markersize': 6,
        'linestyle': ':',
        'alpha': 0.8,
        'zorder': 12
    },
    'ud': {
        'marker': 'o',
        'color': '#0d47a1',
        'linewidth': 2,
        'markersize': 6,
        'alpha': 1,
        'zorder': 1  # Higher zorder to draw on top
    },
    'subud': {
        'marker': '.',
        'color': '#90caf9',
        'linewidth': 1,
        'markersize': 8,
        'alpha': 0.8,
        'zorder': 5
    },
    'wave': {
        'marker': 'o',
        'color': '#d62728', 
        'linewidth': 2,
        'markersize': 6,
        'alpha': 1,
        'zorder': 1
    },
}

def explore_results(exp_dir):
    results = {}
    results_flattened = []
    for root, dirs, files in os.walk(exp_dir):
        if root.endswith('tensorboard'):
            exp_name = root.split('/')[-3]
            if "Poff" in exp_name:
                prefetch = "off"
            elif "Pon" in exp_name:
                prefetch = "on"
            else:
                prefetch = "unknown"
            if "MMR" in exp_name:
                mmr = exp_name.split("MMR")[1].split("_")[0]
            else:
                mmr = "unknown"
            if "exp_final.json" in files:
                with open(os.path.join(root, "exp_final.json")) as f:
                    data = json.load(f)
                    result, result_flattened = parse_result(data)
                    result["prefetch"] = prefetch
                    for i in range(len(result_flattened)):
                        result_flattened[i].extend([prefetch, mmr])
                    results[exp_name] = result
                    results_flattened += result_flattened
                    print(f"{exp_name} - Complete")
                with open(os.path.join(root, "total.json"), "r") as f:
                    total = json.load(f)
                    tf = np.array(total["T_F"]).max()
                    results[exp_name]["T_F"] = tf
                    for i in range(len(result_flattened)):
                        result_flattened[i].append(tf)
            else:
                print(f"{exp_name} - Incomplete")
    df = pd.DataFrame(results_flattened, columns=["schedule", "GBS", "PP", "DP", "model_size", "dyn_extra_mem_factor", "recomputation", "delay_frac", "bandwidth_frac", "iter_time_min", "perf_model_time", "prefetch", "model_mem_ratio", "T_F"])
    return results, df

def parse_result(raw_result):
    iter_times = raw_result["iter_time"]
    perf_model_times = raw_result["perf_model_time"]
    config = raw_result["config"]
    schedule = config["schedule"]
    GBS = config["GBS"]
    PP = config["PP"]
    DP = config["DP"]
    model_size = config["cdc_exp_tf_block_size"]
    dyn_extra_mem_factor = config["dyn_extra_mem_factor"]
    recomputation = config["recomputation"]
    delay_bandwidth_keys = list(iter_times.keys())
    result = {
        "schedule": schedule,
        "GBS": GBS,
        "PP": PP,
        "DP": DP,
        "model_size": model_size,
        "dyn_extra_mem_factor": dyn_extra_mem_factor,
        "recomputation": recomputation,
        "delay_bandwidth_keys": delay_bandwidth_keys,
        "iter_time_min": [],
        "perf_model_time": []
    }

    result_flattened = []
    for key in delay_bandwidth_keys:
        iter_time_min = min(iter_times[key])
        perf_model_time = perf_model_times[key]
        result["iter_time_min"].append(iter_time_min)
        result["perf_model_time"].append(perf_model_time)
        key_eval = eval(key)
        delay_frac = key_eval[0][0]
        bandwidth_frac = key_eval[1][0]
        result_flattened.append([schedule, GBS, PP, DP, model_size, dyn_extra_mem_factor, recomputation, delay_frac, bandwidth_frac, iter_time_min, perf_model_time])
        #print(f"{key} - {iter_time_min}, {perf_model_time}")
    return result, result_flattened

def _get_values(values_list):
    values = np.asarray(values_list)
    ranks = values[:, 0].argsort().argsort()
    return ranks, values[:, 1]

def plot_lat_bw_delay(df):
    # iter_time_min vs delay_frac, bandwidth_frac
    # vary schedule, PP
    static_schedule = ["1F1B", "ZBH1", "ZBV"]
    df["static"] = df["schedule"].apply(lambda x: "static" if x in static_schedule else "dynamic")
    df["static"].iloc[df["prefetch"] == "on"] = "static_optimized"
    pt = pd.pivot_table(df, index=["model_size", "PP", "delay_frac", "bandwidth_frac"], columns=["static"], aggfunc="min", values="iter_time_min")
    pt["speedup wrt static (%)"] = (pt["static"] - pt["dynamic"]) / pt["static"] * 100
    pt["speedup wrt static_optimized (%)"] = (pt["static_optimized"] - pt["dynamic"]) / pt["static"] * 100
    print(pt)
    pt.to_excel(f"{current_dir}/../data/lat_bw_delay_speedup.xlsx")
    PPs = df["PP"].unique()
    model_sizes = df["model_size"].unique()
    for model_size in model_sizes:
        for PP in PPs:
            df_filtered = df[(df["model_size"] == model_size) & (df["PP"] == PP)]
            iter_time_d = {}
            perf_model_time_d = {}
            iter_time_b = {}
            perf_model_time_b = {}
            has_zero_d = {}
            has_zero_b = {}
            for i in range(len(df_filtered)):
                row = df_filtered.iloc[i]
                schedule = row["schedule"]
                prefetch = row["prefetch"]
                if prefetch == "on":
                    schedule += "_P"
                
                if row["bandwidth_frac"] == 0.0:
                    if row["delay_frac"] == 0.0:
                        if not has_zero_d.get(schedule, False):
                            has_zero_d[schedule] = True
                        else:
                            continue
                    iter_time_d[schedule] = iter_time_d.get(schedule, []) + [[row["delay_frac"], row["iter_time_min"]]]
                    perf_model_time_d[schedule] = perf_model_time_d.get(schedule, []) + [[row["delay_frac"], row["perf_model_time"]]]
                if row["delay_frac"] == 0.0:
                    if row["bandwidth_frac"] == 0.0:
                        if not has_zero_b.get(schedule, False):
                            has_zero_b[schedule] = True
                        else:
                            continue
                    iter_time_b[schedule] = iter_time_b.get(schedule, []) + [[row["bandwidth_frac"], row["iter_time_min"]]]
                    perf_model_time_b[schedule] = perf_model_time_b.get(schedule, []) + [[row["bandwidth_frac"], row["perf_model_time"]]]
            if len(iter_time_d) == 0 or len(iter_time_b) == 0:
                continue
            tf = np.nanmean(df_filtered[df_filtered["schedule"] == "1F1B"]["T_F"])
            print(f"Model Size: {model_size}, PP: {PP}, T_F: {tf}")
            fig, ax = plt.subplots(1, 2, figsize=(6, 4))
            #fig.suptitle(f"Model Size: {model_size}, PP: {PP}")
            ax[0].set_title(f"Latency Delay")
            ax[0].set_xlabel(r"$T_{{lat}} / T_{{F}}$")
            ax[0].set_ylabel("Iteration Time (s)")
            ax[1].text(0.4, 0.8, r"$T_F$ = " + f"{tf:.3f}s", horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
            #keys = ["1F1B", "1F1B_P", "ZBH1", "ZBH1_P", "ZBV", "ZBV_P", "ud", "subud", "wave"]
            keys = ["1F1B_P", "ZBH1_P", "ZBV_P", "ud", "subud", "wave"]
            width = 1 / (len(keys) + 1)  # the width of the bars
            multiplier = - (len(keys) - 1) / 2
            for key in keys:
                key_nP = key.removesuffix("_P")
                ranks, values = _get_values(iter_time_d[key])
                offset = width * multiplier
                hatch = None if key.endswith("_P") else '///'
                ax[0].bar(ranks + offset, values, width, label=name_mapping[key], color=schedule_styles[key_nP]['color'], alpha=schedule_styles[key_nP]['alpha'], hatch=hatch)
                ax[0].scatter(ranks + offset, np.asarray(perf_model_time_d[key])[:, 1], color='black', s=8, marker='^', zorder=10)
                ranks, values = _get_values(iter_time_b[key])
                ax[1].bar(ranks + offset, values, width, label=name_mapping[key], color=schedule_styles[key_nP]['color'], alpha=schedule_styles[key_nP]['alpha'], hatch=hatch)
                ax[1].scatter(ranks + offset, np.asarray(perf_model_time_b[key])[:, 1], color='black', s=8, marker='^', zorder=10)
                if key.endswith("_P"):
                    lwp = 0
                    ranks, values = _get_values(iter_time_d[key_nP])
                    ax[0].bar(ranks + offset, values, width - 2*lwp, label=name_mapping[key_nP], facecolor='none', edgecolor='black', linewidth=1, zorder=5)
                    ranks, values = _get_values(iter_time_b[key_nP])
                    ax[1].bar(ranks + offset, values, width - 2*lwp, label=name_mapping[key_nP], facecolor='none', edgecolor='black', linewidth=1, zorder=5)
                multiplier += 1
            fontsize = 'x-small'
            ax[0].legend(fontsize=fontsize)
            ax[0].set_xticks([0, 1, 2, 3], ["0", "0.5", "1.0", "2.0"])
            ax[1].set_title("Bandwidth Delay")
            ax[1].set_xlabel(r"$T_{{bw}} / T_{{F}}$")
            ax[1].set_ylabel("Iteration Time (s)")
            #ax[1].legend(fontsize=fontsize)
            ax[1].set_xticks([0, 1, 2, 3], ["0", "0.5", "1.0", "2.0"])
            fig.tight_layout()
            fig.savefig(f"{current_dir}/../figs/lat_bw_delay_{model_size}_{PP}.pdf", bbox_inches='tight')

def _generate_latex_table(pt):
    template = r"""
\begin{table*}[]
\begin{tabular}{c|c|c|cccccc}
$T_{lat}/T_F$           & $T_{bw}/T_F$     & \textbf{Case} & \textbf{1F1B} & \textbf{ZBH1} & \textbf{ZBV}   & \textbf{CrossUDSub} & \textbf{CrossUD}    & \textbf{CrossWave}  \\ \hline
"""
    for row in pt.iterrows():
        values = row.values
        is_multi_row = row["case"] == 1
        if is_multi_row:
            row_str = r"\multirow{3}{*}{\textbf{" + f"{values[0]:f}" + r"}} & \multirow{3}{*}{\textbf{" + f"{values[1]:f}" + r"}} &"
        else:
            row_str = r" & &"



def plot_extra_gbs_mem(df):
    df = df.drop(columns=["PP", "DP", "model_size", "perf_model_time", "prefetch"])
    df["iter_time_min"] = df["iter_time_min"] / df["GBS"]
    pt = pd.pivot_table(df, index=["delay_frac", "bandwidth_frac", "GBS", "dyn_extra_mem_factor", "recomputation"], columns=["schedule"], aggfunc="min", values="iter_time_min")
    index = pt.index.to_flat_index()
    cases = [1, 2, 3] * (len(index) // 3)
    index = [(d, b, c) for ((d, b, _, _, _), c) in zip(index, cases)]
    pt.index = pd.MultiIndex.from_tuples(index, names=["delay_frac", "bandwidth_frac", "Case"])
    pt.reset_index(inplace=True)
    name_dict = {"delay_frac": "T_lat/T_F", "bandwidth_frac": "T_bw/T_F","1F1B": "1F1B", "ZBH1": "ZBH1", "ZBV": "ZBV", "ud": "CrossUD", "subud": "CrossUDSub", "wave": "CrossWave"}
    pt = pt.rename(columns=name_dict)
    print(pt)
    return pt

def plot_pp_dp_tradeoff(df):
    df["iter_time_min"] = df["iter_time_min"] #/ df["GBS"]
    df = df.drop(columns=["DP", "model_size", "perf_model_time", "prefetch", "dyn_extra_mem_factor", "recomputation"])
    pt = pd.pivot_table(df, index=["model_mem_ratio", "delay_frac", "bandwidth_frac", "GBS", "PP"], columns=["schedule"], aggfunc="min", values="iter_time_min")
    static_schedule = ["1F1B", "ZBH1", "ZBV"]
    chunk1 = pt[static_schedule].loc[(slice(None), slice(None), 2.0, slice(None), slice(None)), :]
    chunk2 = pt[static_schedule].loc[(slice(None), slice(None), 0.25, slice(None), slice(None)), :]
    for j in range(1, len(chunk1)):
        chunk1.iloc[j] = chunk1.iloc[0]
    for j in range(1, len(chunk2)):
        chunk2.iloc[j] = chunk2.iloc[0]
    pt.update(chunk1)
    pt.update(chunk2)
    for mmr in ["0.25", "0.5", "0.75"]:
        for bandwidth_frac in [0.25, 2.0]:
            pt_plot = pt.loc[(mmr, slice(None), bandwidth_frac, slice(None), slice(None)), :]
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            #ax.set_title(f"Model Mem Ratio: {mmr}, Bandwidth Fraction: {bandwidth_frac}")
            ax.set_xlabel("PP")
            ax.set_ylabel("Iteration Time (s)")
            keys = ["1F1B", "ZBH1", "ZBV", "ud", "subud", "wave"]
            hatch_keys = ["ud", "subud", "wave"]
            width = 1 / (len(keys) + 1)  # the width of the bars
            multiplier = - (len(keys) - 1) / 2
            max_value = 0
            for key in keys:
                hatch = None if key not in hatch_keys else '///'
                values = pt_plot[key].values
                max_value = max(max_value, np.max(values))
                ranks = np.array(range(len(values)))
                PPs = pt_plot.index.get_level_values("PP")
                offset = width * multiplier
                ax.bar(ranks + offset, values, width, label=name_mapping[key], color=schedule_styles[key]['color'], alpha=schedule_styles[key]['alpha'], hatch=hatch)
                multiplier += 1
            ax.set_ylim(0, max_value * 1.3)
            ax.legend(fontsize='x-small', ncol=2, loc='upper left')
            ax.set_xticks(list(range(len(values))), PPs)
            fig.tight_layout()
            fig.savefig(f"{current_dir}/../figs/pp_dp_tradeoff_{mmr}_{bandwidth_frac}.pdf", bbox_inches='tight')
    


exps = ['lat_bw_delay', 'extra_gbs_mem', 'dc4', 'pp_dp_tradeoff']
# exps = ['extra_gbs_mem', 'dc4', 'pp_dp_tradeoff']
# mkdir for figs and data
os.makedirs(f"{current_dir}/../figs", exist_ok=True)
os.makedirs(f"{current_dir}/../data", exist_ok=True)
for exp in exps:
    print (f"==============Exploring {exp}==============")
    exp_dir = os.path.join(current_dir, '..', 'clariden', exp) # clariden or clariden_new (Jan 13th 2025)
    results, df = explore_results(exp_dir)
    if exp == "lat_bw_delay":
        plot_lat_bw_delay(df)
    elif exp == "extra_gbs_mem" or exp == "dc4":
        pt = plot_extra_gbs_mem(df)
        pt.to_excel(f"{current_dir}/../data/{exp}.xlsx", float_format="%.3f", na_rep="-")
    elif exp == "pp_dp_tradeoff":
        plot_pp_dp_tradeoff(df)


    df.to_excel(f"{current_dir}/../data/{exp}_raw.xlsx")
    #print(df)