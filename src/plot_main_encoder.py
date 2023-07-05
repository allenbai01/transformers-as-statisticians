from plot_utils import *
from plot_main_gpt import *
import matplotlib.pyplot as plt


def make_ridge_mtl_plot_encoder(metrics, errs_1=None, errs_2=None, ms_square=6, ms_triangle=7):
    # Plot ridge MTL results for the encoder architecture
    plot_path = "../figures"
    figsize = (3.6, 2.7)

    rd1 = {
        "run_name": "nlr_d=20_normalize_noise=0.1",
        "task": "noisy_linear_regression_noise=0.1",
        "model": "EncoderTF",
        "name": "TF_train_noise=0.1",
        "ms_tt": ms_triangle,
    }
    rd11 = {
        "run_name": "nlr_d=20_normalize_noise=0.5",
        "task": "noisy_linear_regression_noise=0.1",
        "model": "EncoderTF",
        "name": "TF_train_noise=0.5",
        "ms_tt": ms_triangle,
    }
    rd2 = {
        "run_name": "nlr_d=20_normalize_noise=0.1",
        "task": "noisy_linear_regression_noise=0.1",
        "model": "ridge_lam=0.2",
        "name": "ridge_lam=0.2",
        "ls": "dotted",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
        "zorder": 1.9,
    }
    rd3 = {
        "run_name": "nlr_d=20_normalize_noise=0.1",
        "task": "noisy_linear_regression_noise=0.1",
        "model": "ridge_lam=5",
        "name": "ridge_lam=5",
        "ls": "dashed",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
        "zorder": 1.9,
    }
    rd4 = {
        "run_name": "nlr_d=20_normalize_mtl",
        "task": "noisy_linear_regression_noise=0.1",
        "model": "EncoderTF",
        "name": "TF_MTL",
        "zorder": 3,
        "marker_tt": "*",
    }
    # run_dicts_ridge = [rd4, rd1, rd11, rd2, rd3]
    # plot_main(metrics, run_dicts_ridge, title=None, fig=fig, ax=ax)
    # plt.tight_layout()
    # fn = "encoder_ridge_1.pdf"
    # plt.savefig(os.path.join(plot_path, fn))
    # plt.show()
    # plt.close()

    rd01 = {
        "run_name": "nlr_d=20_normalize_noise=0.5",
        "task": "noisy_linear_regression_noise=0.5",
        "model": "EncoderTF",
        "name": "TF_train_noise=0.5",
        "ms_tt": ms_triangle,
    }
    rd011 = {
        "run_name": "nlr_d=20_normalize_noise=0.1",
        "task": "noisy_linear_regression_noise=0.5",
        "model": "EncoderTF",
        "name": "TF_train_noise=0.1",
        "ms_tt": ms_triangle,
    }
    rd02 = {
        "run_name": "nlr_d=20_normalize_noise=0.5",
        "task": "noisy_linear_regression_noise=0.5",
        "model": "ridge_lam=0.2",
        "name": "ridge_lam=0.2",
        "ls": "dotted",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
        "zorder": 1.9,
    }
    rd03 = {
        "run_name": "nlr_d=20_normalize_noise=0.5",
        "task": "noisy_linear_regression_noise=0.5",
        "model": "ridge_lam=5",
        "name": "ridge_lam=5",
        "ls": "dashed",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
        "zorder": 1.9,
    }
    rd04 = {
        "run_name": "nlr_d=20_normalize_mtl",
        "task": "noisy_linear_regression_noise=0.5",
        "model": "EncoderTF",
        "name": "TF_MTL",
        "zorder": 3,
        "marker_tt": "*",
    }
    # run_dicts_ridge = [rd04, rd011, rd01, rd02, rd03]
    # plot_main(metrics, run_dicts_ridge, title=None, ylabel=None, fig=fig, ax=ax)
    # plt.tight_layout()
    # fn = "encoder_ridge_2.pdf"
    # plt.savefig(os.path.join(plot_path, fn))
    # plt.show()
    # plt.close()

    # Figure 3
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    run_dicts_two_tasks = [
        ("TF_alg_select", rd4, rd04),
        ("TF_noise_1", rd1, rd011),
        ("TF_noise_2", rd11, rd01),
        ("ridge_lam_1", rd2, rd02),
        ("ridge_lam_2", rd3, rd03),
    ]

    be_x = errs_1[0]
    be_y = errs_2[-1]
    # xlim, ylim = (0.015, 0.105), (0.4, 0.65)
    xlim, ylim = (0.09, 0.31), (0.6, 1.525)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plot_two_tasks(metrics, run_dicts_two_tasks, inds=[20],
                   task_1_name="noisy_reg_noise_1", task_2_name="noisy_reg_noise_2",
                   fig=fig, ax=ax,
                   title=None, error_bar=False)

    ax.plot(errs_1, errs_2, color='k', zorder=1, ls='-', lw=1, label="ridge analytical",
            marker="d", ms=2)
    ax.plot([be_x, be_x], ylim, color='k', zorder=1,
            ls='--', lw=1, alpha=0.7, label="Bayes_err_noise_1")
    ax.plot(xlim, [be_y, be_y], color='k', zorder=1,
            ls='-.', lw=1, alpha=0.7, label="Bayes_err_noise_2")

    # plot_two_tasks(metrics, run_dicts_two_tasks, inds=[20],
    #                task_1_name="=0.1", task_2_name="nlr_noise=0.5",
    #                fig=fig, ax=ax,
    #                title=None)
    #
    # ax.plot([be_x, be_x], ylim, color='k', zorder=1,
    #         ls='--', lw=1, alpha=0.7, label="Bayes_noise_1")
    # ax.plot(xlim, [be_y, be_y], color='k', zorder=1,
    #         ls='-.', lw=1, alpha=0.7, label="Bayes_noise_2")
    # ax.plot(errs_1, errs_2, color='k', zorder=1, ls='-', lw=1, label="ridge analytical",
    #         marker="s", ms=2)
    ax.legend(loc="best", fontsize="small")

    plt.tight_layout()

    fn = "encoder_ridge.pdf"
    plt.savefig(os.path.join(plot_path, fn))
    plt.show()
    plt.close()


def make_linear_logistic_plot_encoder(metrics, ms_square=6, ms_triangle=7):
    # Plot linear + logistic results for the GPT2 architecture
    plot_path = "../figures"
    figsize = (3.6, 2.7)

    # Figure 1
    # fig, ax = plt.subplots(1, 1, figsize=figsize)
    rd1 = {
        "run_name": "linear_and_logistic_d=20",
        "task": "linear_regression",
        "model": "EncoderTF",
        "name": "TF_LR_LC",
        "marker_tt": "*",
    }
    rd21 = {
        "run_name": "linear_classification",
        "task": "linear_regression",
        "model": "EncoderTF",
        "name": "TF_LC",
        "ms_tt": ms_triangle,
    }
    rd22 = {
        "run_name": "linear_regression",
        "task": "linear_regression",
        "model": "EncoderTF",
        "name": "TF_LR",
        "ms_tt": ms_triangle,
    }
    rd2 = {
        "run_name": "linear_and_logistic_d=20",
        "task": "linear_regression",
        "model": "Least Squares",
        "name": "Least Squares",
        "ls": "dotted",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
        "zorder": 1.9,
    }
    rd23 = {
        "run_name": "linear_and_logistic_d=20",
        "task": "linear_regression",
        "model": "Averaging",
        "name": "Averaging",
        "ls": "dashed",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
        "zorder": 1.9,
    }
    rd24 = {
        "run_name": "linear_regression",
        "task": "linear_regression",
        "model": "3-Nearest Neighbors",
        "name": "3-NN",
        "ls": "dashed",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
        "zorder": 1.9,
    }

    # run_dicts_lin = [rd1, rd21, rd22, rd2, rd24]
    # plot_main(metrics, run_dicts_lin, title=None, fig=fig, ax=ax, ylim=(1e-4, 1e2), logscale=True)
    # plt.tight_layout()
    # fn = "encoder_linlog_1.pdf"
    # plt.savefig(os.path.join(plot_path, fn))
    # plt.show()
    # plt.close()

    # Figure 2
    # fig, ax = plt.subplots(1, 1, figsize=figsize)
    rd3 = {
        "run_name": "linear_and_logistic_d=20",
        "task": "linear_classification",
        "model": "EncoderTF",
        "name": "TF_LR_LC",
        "marker_tt": "*",
    }
    rd4 = {
        "run_name": "linear_classification",
        "task": "linear_classification",
        "model": "EncoderTF",
        "name": "TF_LC",
        "ms_tt": ms_triangle,
    }
    rd41 = {
        "run_name": "linear_regression",
        "task": "linear_classification",
        "model": "EncoderTF",
        "name": "TF_LR",
        "ms_tt": ms_triangle,
    }
    rd55 = {
        "run_name": "linear_and_logistic_d=20",
        "task": "linear_classification",
        "model": "Least Squares",
        "name": "Least Squares",
        "ls": "dashed",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
        "zorder": 1.9,
    }
    rd5 = {
        "run_name": "linear_and_logistic_d=20",
        "task": "linear_classification",
        "model": "Averaging",
        "name": "Averaging",
        "ls": "dashed",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
        "zorder": 1.9,
    }
    rd52 = {
        "run_name": "linear_classification",
        "task": "linear_classification",
        "model": "3-Nearest Neighbors",
        "name": "3-NN",
        "ls": "dashed",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
        "zorder": 1.9,
    }
    # run_dicts_log = [rd3, rd4, rd41, rd5, rd52]
    # plot_main(metrics, run_dicts_log, title=None, ylabel="error", fig=fig, ax=ax,
    #           ylim=(0, 0.4), flip_y=True)
    # plt.tight_layout()
    # fn = "encoder_linlog_2.pdf"
    # plt.savefig(os.path.join(plot_path, fn))
    # plt.show()
    # plt.close()

    # Figure 3
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    run_dicts_two_tasks = [
        ("TF_alg_select", rd1, rd3),
        ("TF_reg", rd22, rd41),
        ("TF_cls", rd21, rd4),
        ("Least Squares", rd2, rd55),
        ("Averaging", rd23, rd5),
        ("3-NN", rd24, rd52),
    ]
    xlim = (-0.1, 1.9)
    ylim = (0.165, 0.36)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plot_two_tasks(metrics, run_dicts_two_tasks, inds=[40],
                   task_1_name="regression_square_loss", task_2_name="classification_error",
                   fig=fig, ax=ax, logscale_x=False, flip_y=True,
                   title=None, error_bar=False)
    ax.plot([-0.1, 1.75], [0.195, 0.165], lw=1, ls='--', color='k')
    plt.tight_layout()
    fn = "encoder_linlog.pdf"
    plt.savefig(os.path.join(plot_path, fn))
    plt.show()
    plt.close()
