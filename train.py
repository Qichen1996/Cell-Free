#!/usr/bin/env python
import sys
import os
import wandb
import torch
import socket
import numpy as np
from arguments import *
from env import MultiCellNetEnv
from utils import *
from env.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

# 强制覆盖 WandB 使用的账号和密钥
os.environ["WANDB_API_KEY"] = "58aa2e20bdb9bc77a74e030c3af1f4249a2b4f6e"
os.environ["WANDB_ENTITY"] = "keyuli207-kth-royal-institute-of-technology"

# 可选：防止 WandB 把代码全上传
os.environ["WANDB_DISABLE_CODE"] = "true"


if DEBUG and 'loadDebugger' in globals():
    loadDebugger()
    
def get_env_kwargs(args):
    return {k: v for k, v in vars(args).items() if v is not None}

def get_default_env_config(args, env_args):
    tmp_env = MultiCellNetEnv(**get_env_kwargs(env_args))
    tmp_env.print_info()
    # tmp_env.net.traffic_model.print_info()
    args.__dict__.update(
        episode_length=tmp_env.episode_len // args.n_rollout_threads,
        episode_secs=tmp_env.episode_time_len,
        avg_traffic_density=tmp_env.net.traffic_model.density_mean,
        traffic_density_std=tmp_env.net.traffic_model.density_std,
        accelerate=tmp_env.net.accelerate,
        # w_pc=tmp_env.w_pc,
        w_qos=tmp_env.w_qos,
        w_xqos=tmp_env.w_xqos,
        # w_drop=tmp_env.w_drop,
        # w_delay=tmp_env.w_delay,
    )

def make_env(args, env_args, for_eval=False):
    n_threads = args.n_rollout_threads

    def get_env_fn(rank):
        def init_env():
            kwargs = get_env_kwargs(env_args)
            kwargs['start_time'] = rank / n_threads * args.episode_secs
            kwargs['episode_len'] = args.episode_length
            env = MultiCellNetEnv(**kwargs)
            if for_eval:
                env.seed(args.seed * 50000 + rank * 10000)
            else:
                env.seed(args.seed + rank * 1000)
            return env
        return init_env
    
    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def main(args):
    parser = get_config()
    env_parser = get_env_config()
    args, env_args = parser.parse_known_args(args)
    env_args, rl_args = env_parser.parse_known_args(env_args)
    
    if args.algorithm_name == "rmappo":
        assert (args.use_recurrent_policy or args.use_naive_recurrent_policy), (
            "check recurrent policy!")
        from trainers.mappo_trainer import MappoTrainer as Trainer, get_mappo_config
        rl_parser = get_mappo_config()
    elif args.algorithm_name == "mappo":
        args.use_recurrent_policy = False
        args.use_naive_recurrent_policy = False
        from trainers.mappo_trainer import MappoTrainer as Trainer, get_mappo_config
        rl_parser = get_mappo_config()
    elif args.algorithm_name == "dqn":
        from trainers.dqn_trainer import DQNTrainer as Trainer, get_dqn_config
        rl_parser = get_dqn_config()
    else:
        raise NotImplementedError
    
    rl_args = rl_parser.parse_args(rl_args)
    vars(args).update(vars(rl_args))

    # cuda
    if args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)

    set_log_level(args.log_level)

    if args.sim_log_path is None:
        fn = '{}_{}_{}.log'.format(
            env_args.scenario, args.algorithm_name, args.experiment_name)
        args.sim_log_path = 'logs/' + fn

    set_log_file(args.sim_log_path)
    
    # get env config
    get_default_env_config(args, env_args)
    
    # run dir
    run_dir = get_run_dir(args, env_args)
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # logging
    if args.use_wandb:
        wandb.init(
            config=args,
            project="cellfree",
            entity="keyuli207-kth-royal-institute-of-technology",
            notes=socket.gethostname(),
            name=f"{args.algorithm_name}_{args.experiment_name}_seed{args.seed}",
            group=env_args.scenario,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                             run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # env init
    envs = make_env(args, env_args)
    eval_envs = make_env(args, env_args, for_eval=True) if args.use_eval else None

    config = {
        "all_args": args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": MultiCellNetEnv.num_agents,
        "device": device,
        "run_dir": run_dir
    }

    trainer = Trainer(config)
    trainer.train()
    trainer.close()


def runtime(run_time):
    hours = int(run_time // 3600)
    minutes = int((run_time % 3600) // 60)
    seconds = int(run_time % 60)
    print(int(run_time))
    print(f"程序运行时间：{hours}小时 {minutes}分钟 {seconds}秒")

if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtime(elapsed_time)