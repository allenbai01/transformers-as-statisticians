import os

import matplotlib
matplotlib.rcParams['axes.linewidth'] = 0.5
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from eval import get_run_metrics, baseline_names, get_model_from_run
from models import build_model

import pdb

# sns.set_theme("notebook", "darkgrid")
# palette = sns.color_palette("colorblind")
palette = sns.color_palette("tab10")


relevant_model_names = {
    "linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
    ],
    "sparse_linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
        "Lasso (alpha=0.01)",
    ],
    "decision_tree": [
        "Transformer",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
        "Greedy Tree Learning",
        "XGBoost",
    ],
    "relu_2nn_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
    ],
    "linear_classification": [
        "Transformer",
        "3-Nearest Neighbors",
        "Averaging",
    ],
}


def basic_plot(metrics, models=None, trivial=1.0, ylabel="squared error", fig=None, ax=None):
    if fig is None:
        fig, ax = plt.subplots(1, 1)

    if models is not None:
        metrics = {k: metrics[k] for k in models}

    color = 0
    if trivial is not None:
        ax.axhline(trivial, ls="--", color="gray")
    for name, vs in metrics.items():
        ax.plot(vs["mean"], "-", label=name, color=palette[color % 10], lw=2)
        low = [x - y / np.sqrt(20) for (x, y) in zip(vs["mean"], vs["std"])]
        high = [x + y / np.sqrt(20) for (x, y) in zip(vs["mean"], vs["std"])]
        # low = vs["bootstrap_low"]
        # high = vs["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, alpha=0.3)
        color += 1
    ax.set_xlabel("in-context examples")
    ax.set_ylabel(ylabel)
    # ax.set_ylabel("squared error")
    ax.set_xlim(-1, len(low) + 0.1)
    # ax.set_ylim(-0.1, 1.25)

    # legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    legend = ax.legend(loc="best", fontsize="small")
    if fig is None:
        fig.set_size_inches(4, 3)
        for line in legend.get_lines():
            line.set_linewidth(3)


    return fig, ax


def plot_main(all_metrics, run_dicts, ylabel="squared error", title=None, fig=None, ax=None,
              logscale=False, ylim=None, flip_y=False, sample_size=6400, legend_loc="best", legend_size="small"):
    if fig is None:
        fig, ax = plt.subplots(1, 1)

    color = 0
    for rd in run_dicts:
        vs = all_metrics[rd["run_name"]][rd["task"]][rd["model"]]

        try:
            curr_color = rd["color"]
        except:
            curr_color = palette[color % 10]
            color += 1

        bold_font = False
        try:
            ls = rd["ls"]
            zorder = 1
        except:
            ls = "solid"
            try:
                zorder = rd["zorder"]
                if zorder == 3:
                    bold_font = True
            except:
                zorder = 2

        try:
            marker = rd["marker"]
        except:
            marker = "d"

        mean, std = np.array(vs["mean"]), np.array(vs["std"]) / np.sqrt(sample_size)
        low, high = mean - std, mean + std
        # low, high = np.array(vs["bootstrap_low"]), np.array(vs["bootstrap_high"])
        # low = [x - y / np.sqrt(20) for (x, y) in zip(vs["mean"], vs["std"])]
        # high = [x + y / np.sqrt(20) for (x, y) in zip(vs["mean"], vs["std"])]
        if flip_y:
            mean, low, high = 1 - mean, 1 - low, 1 - high

        ax.plot(mean, "-", label=rd["name"], color=curr_color, ls=ls, lw=1,
                marker=marker, ms=3, markevery=2,
                zorder=zorder)
        if ls == "solid":
            ax.fill_between(range(len(low)), low, high, alpha=0.3, color=curr_color)

    # styles
    ax.set_facecolor('whitesmoke')
    ax.set_axisbelow(True)
    ax.tick_params(direction='in')
    ax.grid(lw=0.25, color='0.5')

    ax.set_xlabel("in-context examples", fontsize="small")
    ax.set_ylabel(ylabel, fontsize="small")
    # ax.set_ylabel("squared error")
    ax.set_xlim(-1, len(low) + 0.1)
    # ax.set_ylim(-0.1, 1.25)
    if ylim:
        ax.set_ylim(ylim)
    if logscale:
        ax.set_yscale("log")

    # legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # pdb.set_trace()

    ax.legend(loc=legend_loc, fontsize=legend_size, framealpha=0.7)
    # for text in ax.legend().get_texts():
    #     if "+" in text.get_text():
    #         text.set_color("blue")

    if title is not None:
        ax.set_title(title, fontsize="small")

def plot_two_tasks(
        all_metrics, run_dicts, inds=[10, 40],
        task_1_name="linear_regression", task_2_name="linear_regression",
        title=None, fig=None, ax=None,
        logscale_x=False, logscale_y=False, ylim=None, flip_y=False,
        sample_size=6400,
        error_bar=True,
        legend_size="small",
):
    if fig is None:
        fig, ax = plt.subplots(1, 1)

    color = 0
    for ind in inds:
        for (alg_name, rd1, rd2) in run_dicts:
            vs1 = all_metrics[rd1["run_name"]][rd1["task"]][rd1["model"]]
            vs2 = all_metrics[rd2["run_name"]][rd2["task"]][rd2["model"]]

            # try:
            #     marker = rd1["marker_tt"]
            # except:
            #     marker = "^"

            # try:
            #     ms = rd1["ms_tt"]
            # except:
            #     ms = 10

            (marker, ms) = name_to_marker(alg_name)

            try:
                zorder = rd1["zorder"]
            except:
                zorder = 2

            x, y = vs1["mean"][ind], vs2["mean"][ind]
            x_std, y_std = vs1["std"][ind] / np.sqrt(sample_size), vs2["std"][ind] / np.sqrt(sample_size)
            # bar_low, bar_high = vs2["mean"][ind] - vs2["bootstrap_low"][ind], vs2["bootstrap_high"][ind] - vs2["mean"][ind]
            if flip_y:
                y = 1 - y
                # bar_low, bar_high = bar_high, bar_low

            if error_bar:
                ax.errorbar(
                    x, y, xerr=x_std, yerr=y_std,
                    mec='k', mew=.5, mfc=palette[color % 10],
                    marker=marker, ms=ms, lw=1, ls="",
                    label=alg_name, zorder=zorder,
                )
            else:
                ax.plot(
                    x, y, mfc=palette[color % 10],
                    marker=marker, mec='k', mew=.5, ms=ms, lw=1, ls="",
                    label=alg_name, zorder=zorder,
                )

            # ax.plot(vs1["mean"][ind], vs2["mean"][ind], color=palette[color % 10],
            #         marker='d', markersize=10, ls='',
            #         label=alg_name)

            # ax.plot(vs["mean"], "-", label=rd["name"], color=palette[color % 10], lw=2)
            # low = vs["bootstrap_low"]
            # high = vs["bootstrap_high"]
            # ax.fill_between(range(len(low)), low, high, alpha=0.3)
            color += 1

    # styles
    ax.set_facecolor('whitesmoke')
    ax.set_axisbelow(True)
    ax.tick_params(direction='in')
    ax.grid(lw=0.25, color='0.5')

    ax.set_xlabel(task_1_name, fontsize="small")
    ax.set_ylabel(task_2_name, fontsize="small")
    # ax.set_ylabel("squared error")
    # ax.set_xlim(-1, len(low) + 0.1)
    # ax.set_ylim(-0.1, 1.25)
    if ylim:
        ax.set_ylim(ylim)
    if logscale_x:
        ax.set_xscale("log")
    if logscale_y:
        ax.set_yscale("log")

    # legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.legend(loc="best", fontsize=legend_size, framealpha=0.7)

    if title is not None:
        ax.set_title(title, fontsize="small")


def get_relevant_models_for_bar_plot(problem):
    if problem == "linear_regression":
        return [
            ("EncoderTF", "Transformer"),
            ("Least Squares", "Least Squares"),
            ("Averaging", "Averaging"),
            ("3-Nearest Neighbors", "3-NN"),
        ]
    elif problem == "sparse_linear_regression":
        return [
            ("EncoderTF", "Transformer"),
            ("Least Squares", "Least Squares"),
            ("Lasso (alpha=1)", "Lasso_lam=1"),
            ("Lasso (alpha=0.1)", "Lasso_lam=0.1"),
            ("Lasso (alpha=0.01)", "Lasso_lam=0.01"),
            ("Lasso (alpha=0.001)", "Lasso_lam=0.001"),
        ]
    elif "noisy_linear_regression" in problem:
        return [
            ("EncoderTF", "Transformer"),
            ("ridge_lam=0.2", "ridge_lam_1"),
            ("ridge_lam=5", "ridge_lam_2"),
            ("Averaging", "Averaging"),
            ("3-Nearest Neighbors", "3-NN"),
        ]
    elif problem == "linear_classification":
        return [
            ("EncoderTF", "Transformer"),
            ("Averaging", "Averaging"),
            ("3-Nearest Neighbors", "3-NN"),
            ("logistic_regression", "Logistic Regression"),
            # ("Least Squares", "Least Squares"),
        ]


def name_to_marker(name):
    if "Transformer" in name or "TF" in name:
        return ("*", 10)
    elif name == "Least Squares":
        return ("^", 8)
    elif name == "Logistic Regression":
        return ("v", 8)
    elif name in ["Averaging", "3-NN"]:
        return ("s", 7)
    elif "ridge_lam" in name:
        return ("D", 6)
    elif name == "ridge analytical":
        return ("d", 2)
    elif "Lasso" in name:
        return ("h", 8)

    return ("", 0)


def plot_bar(
        metrics, problems, run_names, problem_labels=None, title=None, fig=None, ax=None,
        inds=None, figsize=(6, 3), figsize_st=(3.6, 2.7),
        flip_problems=[],
        sample_size=6400, error_bar=True, alpha=0.85,
):
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # style
    ax.set_facecolor('whitesmoke')
    ax.set_axisbelow(True)
    # ax.tick_params(direction='in')
    ax.grid(lw=0.25, color='0.5')

    y_pos = np.arange(len(problems))

    if inds is None:
        inds = [40 for _ in range(len(problems))]

    color = 0
    color_by_names = {}

    if problem_labels == None:
        problem_labels = problems

    for (problem, run_name, y, ind) in zip(problems, run_names, y_pos, inds):

        models_and_names = get_relevant_models_for_bar_plot(problem)

        for i, (model, name) in enumerate(models_and_names):
            vs = metrics[run_name][problem][model]

            marker, ms = name_to_marker(name)

            zorder = 3 if model == "EncoderTF" else 2

            x = vs["mean"][ind]
            x_low_width, x_high_width = vs["std"][ind] / np.sqrt(sample_size), vs["std"][ind] / np.sqrt(sample_size)
            # vs["mean"][ind] - vs["bootstrap_low"][ind], vs["bootstrap_high"][ind] - vs["mean"][ind]

            if problem in flip_problems:
                x = 1 - x
                x_low_width, x_high_width = x_high_width, x_low_width

            if name not in color_by_names.keys():
                label = name
                if name == "Transformer":
                    color_curr = "gray"
                else:
                    color_curr = palette[color % 10]
                    color += 1
                color_by_names[name] = color_curr
            else:
                label = None
                color_curr = color_by_names[name]

            mew = .5 if name == "Transformer" else .5

            if error_bar:
                ax.errorbar(
                    np.array([[ x ]]),
                    y,
                    xerr=np.array(
                        [[x_low_width], [x_high_width]]),
                    yerr=None,
                    mec='k', mew=mew, mfc=color_curr,
                    marker=marker, ms=ms, lw=1, ls="", zorder=zorder,
                    alpha=alpha,
                    label=label,
                )
            else:
                ax.plot(x, y, marker=marker, ms=ms, lw=1, ls="", zorder=zorder,
                        alpha=alpha, label=label, mec='k', mew=mew, mfc=color_curr)

        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="smaller")

    ax.set_yticks(y_pos, labels=problem_labels)
    ax.invert_yaxis()  # labels read top-to-bottom

    if title is not None:
        ax.set_title(title, fontsize="small")

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="smaller")
    ax.set_xlabel("Loss", fontsize="small")
    # ax.legend(loc="best", fontsize="small")
    fig.tight_layout()

    path = "../figures/encoder_basic.pdf"
    fig.savefig(path)


def collect_results(run_dir, df, valid_row=None, rename_eval=None, rename_model=None):
    all_metrics = {}
    for _, r in df.iterrows():
        if valid_row is not None and not valid_row(r):
            continue

        run_path = os.path.join(run_dir, r.task, r.run_id)
        _, conf = get_model_from_run(run_path, only_conf=True)

        print(r.run_name, r.run_id)
        metrics = get_run_metrics(run_path, skip_model_load=True)

        for eval_name, results in sorted(metrics.items()):
            processed_results = {}
            for model_name, m in results.items():
                if "gpt2" in model_name in model_name:
                    model_name = r.model
                    if rename_model is not None:
                        model_name = rename_model(model_name, r)
                else:
                    model_name = baseline_names(model_name)
                m_processed = {}
                n_dims = conf.model.n_dims

                # pdb.set_trace()

                xlim = conf.training.curriculum.points.end
                normalization = 1
                # if r.task in ["relu_2nn_regression", "decision_tree"]:
                #     xlim = 200
                #
                # normalization = n_dims
                # if r.task == "sparse_linear_regression":
                #     normalization = int(r.kwargs.split("=")[-1])
                # if r.task == "decision_tree":
                #     normalization = 1
                # # if "classification" in r.task:

                # pdb.set_trace()

                for k, v in m.items():
                    v = v[:xlim]
                    v = [vv / normalization for vv in v]
                    m_processed[k] = v
                processed_results[model_name] = m_processed
            if rename_eval is not None:
                eval_name = rename_eval(eval_name, r)
            if eval_name not in all_metrics:
                all_metrics[eval_name] = {}
            all_metrics[eval_name].update(processed_results)
    return all_metrics


def collect_all_results(run_dir, df, rename_eval=None, rename_model=None, step=-1):
    all_metrics = {}
    for _, r in df.iterrows():
        # if valid_row is not None and not valid_row(r):
        #     continue

        run_path = os.path.join(run_dir, r.task, r.run_id)
        _, conf = get_model_from_run(run_path, only_conf=True)

        print(r.run_name, r.run_id)

        # if r.run_name == "slr_d=20_k=3_normalize_guided_lam_0.1_token_5_layer_-2":
        #     pdb.set_trace()

        metrics = get_run_metrics(run_path, skip_model_load=True, step=step)

        all_metrics[r.run_name] = {}

        for eval_name, results in sorted(metrics.items()):
            processed_results = {}
            for model_name, m in results.items():
                # if "gpt2" in model_name in model_name:
                if "gpt2" in model_name or "EncoderTF" in model_name:
                    model_name = r.model
                    if rename_model is not None:
                        model_name = rename_model(model_name, r)
                else:
                    model_name = baseline_names(model_name)
                m_processed = {}
                n_dims = conf.model.n_dims

                # pdb.set_trace()

                xlim = conf.training.curriculum.points.end
                normalization = 1
                # if r.task in ["relu_2nn_regression", "decision_tree"]:
                #     xlim = 200
                #
                # normalization = n_dims
                # if r.task == "sparse_linear_regression":
                #     normalization = int(r.kwargs.split("=")[-1])
                # if r.task == "decision_tree":
                #     normalization = 1
                # # if "classification" in r.task:

                # pdb.set_trace()

                for k, v in m.items():
                    v = v[:xlim]
                    v = [vv / normalization for vv in v]
                    m_processed[k] = v
                processed_results[model_name] = m_processed
            if rename_eval is not None:
                eval_name = rename_eval(eval_name, r)
            # if eval_name not in all_metrics:
            all_metrics[r.run_name][eval_name] = {}
            # pdb.set_trace()
            all_metrics[r.run_name][eval_name].update(processed_results)
    return all_metrics