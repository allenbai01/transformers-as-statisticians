from plot_utils import *


def noisy_lr_ridge_error(d, N, tau=1.0, sigma=1.0, lams=None, n_rep=1000):
    s = 0
    if lams is None:
        # compute optimal lambda corresponding to Bayes optimal ridge
        lams = [sigma**2 / tau**2]
    errs = np.zeros_like(lams)
    for i in range(n_rep):
        X = np.random.randn(N, d)
        XtX = np.dot(X.T, X)
        for (j, lam) in enumerate(lams):
            inv = np.linalg.inv(XtX + lam * np.eye(d))
            err = sigma**2 * (1 + np.trace(inv)) + (lam**2 * tau**2 - lam * sigma**2) * np.sum(inv * inv) # simplified exact formula for squared error of ridge
            errs[j] += err
        if i % 1000 == 0:
            print(i)
    return errs / n_rep


def get_ridge_errors(d=20, N=20):
    tau = 1 / np.sqrt(d)
    lams = np.geomspace(0.2, 5, 20)
    sigma_1, sigma_2 = 0.1, 0.5
    n_rep = 10000
    errs_1 = noisy_lr_ridge_error(d, N, tau=tau, sigma=sigma_1, lams=lams, n_rep=n_rep)
    errs_2 = noisy_lr_ridge_error(d, N, tau=tau, sigma=sigma_2, lams=lams, n_rep=n_rep)
    return errs_1, errs_2


def make_ridge_mtl_plot(metrics, errs_1, errs_2, ms_square=6, ms_triangle=7):
    # Plot ridge MTL results for the GPT2 architecture
    plot_path = "../figures"
    figsize = (3.2, 2.4)
    legend_size = 7

    # Figure 1
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    rd1 = {
        "run_name": "nlr_d=20_normalize_noise=0.1",
        "task": "noisy_linear_regression_noise=0.1",
        "model": "Transformer",
        "name": r"TF_noise_1",
        "ms_tt": ms_triangle,
    }
    rd11 = {
        "run_name": "nlr_d=20_normalize_noise=0.5",
        "task": "noisy_linear_regression_noise=0.1",
        "model": "Transformer",
        "name": r"TF_noise_2",
        "ms_tt": ms_triangle,
    }
    rd2 = {
        "run_name": "nlr_d=20_normalize_noise=0.1",
        "task": "noisy_linear_regression_noise=0.1",
        "model": "ridge_lam=0.2",
        "name": "ridge_lam_1",
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
        "name": "ridge_lam_2",
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
        "model": "Transformer",
        "name": r"TF_alg_select",
        "zorder": 3,
        "marker_tt": "*",
    }
    run_dicts_ridge = [rd4, rd1, rd11, rd2, rd3]
    plot_main(metrics, run_dicts_ridge, title=None, ylabel="square loss", fig=fig, ax=ax, legend_size=legend_size)
    plt.tight_layout()
    fn = "ridge_1.pdf"
    plt.savefig(os.path.join(plot_path, fn))
    plt.show()
    plt.close()

    # Figure 2
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    rd01 = {
        "run_name": "nlr_d=20_normalize_noise=0.5",
        "task": "noisy_linear_regression_noise=0.5",
        "model": "Transformer",
        "name": r"TF_noise_2",
        "ms_tt": ms_triangle,
    }
    rd011 = {
        "run_name": "nlr_d=20_normalize_noise=0.1",
        "task": "noisy_linear_regression_noise=0.5",
        "model": "Transformer",
        "name": r"TF_noise_1",
        "ms_tt": ms_triangle,
    }
    rd02 = {
        "run_name": "nlr_d=20_normalize_noise=0.5",
        "task": "noisy_linear_regression_noise=0.5",
        "model": "ridge_lam=0.2",
        "name": "ridge_lam_1",
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
        "name": "ridge_lam_2",
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
        "model": "Transformer",
        "name": r"TF_alg_select",
        "zorder": 3,
        "marker_tt": "*",
    }
    run_dicts_ridge = [rd04, rd011, rd01, rd02, rd03]
    plot_main(metrics, run_dicts_ridge, title=None, ylabel=" ", fig=fig, ax=ax, legend_size=legend_size)
    plt.tight_layout()
    fn = "ridge_2.pdf"
    plt.savefig(os.path.join(plot_path, fn))
    plt.show()
    plt.close()

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

    ax.legend(loc="best", fontsize=legend_size, framealpha=0.7)

    plt.tight_layout()

    fn = "ridge_3.pdf"
    plt.savefig(os.path.join(plot_path, fn))
    plt.show()
    plt.close()


def make_linear_logistic_plot(metrics, ms_square=6, ms_triangle=7):
    # Plot linear + logistic results for the GPT2 architecture
    plot_path = "../figures"
    figsize = (3.2, 2.4)
    legend_size = 7


    # Figure 1
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    rd1 = {
        "run_name": "linear_and_logistic_d=20",
        "task": "linear_regression",
        "model": "Transformer",
        "name": "TF_alg_select",
        "marker_tt": "*",
        "zorder": 3,
    }
    rd21 = {
        "run_name": "linear_classification",
        "task": "linear_regression",
        "model": "Transformer",
        "name": "TF_cls",
        "ms_tt": ms_triangle,
    }
    rd22 = {
        "run_name": "linear_regression",
        "task": "linear_regression",
        "model": "Transformer",
        "name": "TF_reg",
        "ms_tt": ms_triangle,
    }
    rd2 = {
        "run_name": "linear_and_logistic_d=20",
        "task": "linear_regression",
        "model": "Least Squares",
        "name": "Least Squares",
        "ls": "--",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
    }
    rd23 = {
        "run_name": "linear_and_logistic_d=20",
        "task": "linear_regression",
        "model": "Averaging",
        "name": "Averaging",
        "ls": "dotted",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
    }
    rd52 = {
        "run_name": "linear_regression",
        "task": "linear_regression",
        "model": "3-Nearest Neighbors",
        "name": "3-NN",
        "ls": "dotted",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
    }

    run_dicts_lin = [rd1, rd22, rd21, rd2]
    plot_main(metrics, run_dicts_lin, title=None, ylabel="square loss", fig=fig, ax=ax, ylim=(-0.05, 1.05),
              logscale=False, legend_loc="upper right", legend_size=legend_size)
    plt.tight_layout()
    fn = "linlog_1.pdf"
    plt.savefig(os.path.join(plot_path, fn))
    plt.show()
    plt.close()

    # Figure 2
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    rd3 = {
        "run_name": "linear_and_logistic_d=20",
        "task": "linear_classification",
        "model": "Transformer",
        "name": "TF_alg_select",
        "marker_tt": "*",
        "zorder": 3,
    }
    rd4 = {
        "run_name": "linear_classification",
        "task": "linear_classification",
        "model": "Transformer",
        "name": "TF_cls",
        "ms_tt": ms_triangle,
    }
    rd41 = {
        "run_name": "linear_regression",
        "task": "linear_classification",
        "model": "Transformer",
        "name": "TF_reg",
        "ms_tt": ms_triangle,
    }
    rd5 = {
        "run_name": "linear_and_logistic_d=20",
        "task": "linear_classification",
        "model": "Least Squares",
        "name": "Least Squares",
        "ls": "dashed",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
    }
    rd53 = {
        "run_name": "linear_classification",
        "task": "linear_classification",
        "model": "3-Nearest Neighbors",
        "name": "3-NN",
        "ls": "dotted",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
    }
    rd24 = {
        "run_name": "linear_classification",
        "task": "linear_classification",
        "model": "Averaging",
        "name": "Averaging",
        "ls": "dotted",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
    }
    rd6 = {
        "run_name": "linear_classification",
        "task": "linear_classification",
        "model": "logistic_regression",
        "name": "Logistic Regression",
        "ls": "--",
        "color": "black",
        "marker": "",
        "marker_tt": "s",
        "ms_tt": ms_square,
    }
    run_dicts_log = [rd3, rd41, rd4, rd6]
    plot_main(metrics, run_dicts_log, title=None, ylabel="error", fig=fig, ax=ax,
              ylim=(0.15, 0.4), flip_y=True, legend_size=legend_size)
    plt.tight_layout()
    fn = "linlog_2.pdf"
    plt.savefig(os.path.join(plot_path, fn))
    plt.show()
    plt.close()

    # Figure 3
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    run_dicts_two_tasks = [
        ("TF_alg_select", rd1, rd3),
        ("TF_reg", rd22, rd41),
        ("TF_cls", rd21, rd4),
        ("Least Squares", rd2, rd5),
        ("Averaging", rd23, rd24),
        ("3-NN", rd52, rd53),
    ]
    # xlim = (-0.05, 4)
    # xlim = (-.1e-2, 4)
    xlim = (-0.4, 4)
    ylim = (0.17, 0.36)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax.set_xscale("function", functions=[lambda x: np.log(x + 1e-1), lambda x: np.exp(x) - 1e-1])
    # ax.set_xticks([0, 1, 2, 4])
    plot_two_tasks(metrics, run_dicts_two_tasks, inds=[40],
                   task_1_name="regression_square_loss", task_2_name="classification_error",
                   fig=fig, ax=ax, logscale_x=False, flip_y=True,
                   title=None, error_bar=False, legend_size=legend_size)
    ax.plot([-0.4, 3.75], [0.195, 0.17], lw=1, ls='--', color='k')
    plt.tight_layout()
    fn = "linlog_3.pdf"
    plt.savefig(os.path.join(plot_path, fn))
    plt.show()
    plt.close()
