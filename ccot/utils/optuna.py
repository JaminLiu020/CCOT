import optuna  #用于优化超参数
import logging  #用于日志记录
import sys
import torch
from ccot.evaluate.evaluate import eval_fxn

def setup_logging(outdir):
    # 创建日志记录器
    log_file = f"{outdir}/optuna_search_log.txt"
    # 创建文件处理器，将日志写入文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器，将日志输出到控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 获取根日志记录器并添加处理器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 确保没有重复的处理器添加
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    # 配置 Optuna 的日志记录
    optuna.logging.enable_propagation()  # 让 Optuna 日志传递到根日志记录器
    optuna.logging.set_verbosity(optuna.logging.INFO)  # 设置日志级别

# 运行超参数优化
def run_hyperparameter_search(config, outdir, train):
    # sigma_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 100000.0, 10000000.0]

    # 定义optuna目标函数，该函数用于在超参数搜索过程中调用
    def objective(trial):
        # 1. 定义超参数的搜索空间
        # beta_values = [5.0, 10.0]
        # beta = trial.suggest_categorical('beta', beta_values)  # 离散的取值

        # sigma_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 100000.0, 10000000.0]
        # sigma = trial.suggest_categorical('sigma', sigma_values)  # 离散的取值
        alpha = trial.suggest_float('alpha', float(config.optuna.alpha.min), float(config.optuna.alpha.max))  # 均匀分布
        # sigma = trial.suggest_float('sigma', float(config.optuna.sigma.min), float(config.optuna.sigma.max), log=True)  # 对数均匀分布

        # 2. 将超参数写入到config中
        config.model.adaptive_mass_transport.alpha = alpha
        # config.model.adaptive_mass_transport.sigma = sigma
        # config.model.classifier_free_guidance.beta = beta

        # 3. 调用训练函数，传递outdir和修改后的config
        try:
            # 4. 使用评估指标作为返回值，optuna 会根据这个值来调整超参数
            eval_metric = train(outdir, config)
        except ValueError as error:
            print("Training failed due to a ValueError")
            return float('inf')  # 训练失败时返回一个较大的损失值

        return eval_metric  # 目标是最小化该评估指标

    savedir = outdir / 'optuna'
    savedir.mkdir(parents=True, exist_ok=True)
    # 设置日志记录
    setup_logging(savedir)

    # 使用 SQLite 数据库保存和加载搜索进度
    study_name = "ccot_sigma_study"  # 用于标识搜索的名称
    storage = f"sqlite:///{savedir}/optuna_search.db"  # 在 outdir 目录下保存数据库文件

    # 如果 study 不存在则创建新的，如果存在则继续
    study = optuna.create_study(study_name=study_name, storage=storage, direction='minimize', load_if_exists=True)

    # # 将每个 sigma 值排入队列
    # for sigma in sigma_values:
    #     study.enqueue_trial({"sigma": sigma})

    ori_model = torch.load(outdir/ 'cache' / 'best_model.pt')
    ori_mmd = ori_model['minmmd']
    logging.info("Starting hyperparameter search")
    logging.info(f'save directory: {savedir}')
    logging.info(f'original model minmmd: {ori_mmd}')
    # logging.info(f'alpha: fixed {config.model.adaptive_mass_transport.alpha}')
    logging.info(f'sigma: fixed {config.model.adaptive_mass_transport.sigma}')
    logging.info(f'beta: fixed {config.model.classifier_free_guidance.beta}')
    # logging.info(f'alpha: [{config.optuna.alpha.min},{config.optuna.alpha.max}]')
    # logging.info(f'sigma: [{config.optuna.sigma.min},{config.optuna.sigma.max}]')
    # logging.info(f'beta: [5.0 and 10.0]')
    logging.info(f'{config.optuna.n_iters_per_search} epochs per search')
    logging.info(f'num_searchs: {config.optuna.num_searchs}')
    study.optimize(objective, n_trials=config.optuna.num_searchs)

    # 输出最优超参数
    logging.info(f"Best hyperparameters: {study.best_params}")
    logging.info(f"Best evaluation metric: {study.best_value}")

    # 保存最优模型
    sigma = config.model.adaptive_mass_transport.sigma
    alpha = config.model.adaptive_mass_transport.alpha
    beta = config.model.classifier_free_guidance.beta

    if 'sigma' in study.best_params.keys():
        sigma = study.best_params['sigma']

    if 'alpha' in study.best_params.keys():
        alpha = study.best_params['alpha']

    if 'beta' in study.best_params.keys():
        beta = study.best_params['beta']

    best_model_name = f"alpha_{alpha}_sigma_{sigma}_beta_{beta}.pt"
    logging.info(f"The best model has been saved as: {savedir}/{best_model_name}")

    del config['optuna']
    evalmmd = eval_fxn(outdir, config, model_name=best_model_name)

    return evalmmd
