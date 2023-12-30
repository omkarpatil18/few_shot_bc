import sys
import torch
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import datetime
from utils import (
    compute_dict_mean,
    set_seed,
    detach_dict,
    save_videos,
)  # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from rl_bench.torch_data import load_data as load_rlbench_data, ReverseTrajDataset
from rlbench.tasks import (
    OpenDoor,
    OpenBox,
    CloseBox,
    CloseDoor,
    ToiletSeatDown,
    ToiletSeatUp,
    OpenGrill,
    CloseGrill,
)
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

sys.path.append("act/act/")
FRANKA_JOINT_LIMITS = np.asarray(
    [
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    ],
    dtype=np.float32,
).T
FORWARD_ENVS = {OpenDoor, OpenBox, ToiletSeatUp, OpenGrill}
BACKWARD_ENVS = {CloseBox, CloseDoor, ToiletSeatDown, CloseGrill}


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args["eval"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size = args["batch_size"]
    num_epochs = args["num_epochs"]
    add_task_ind = args["add_task_ind"]
    ckpt_dir = args["ckpt_dir"]
    ckpt_names = args["ckpt_names"]
    few_shot = args["few_shot"]
    ent_weight = args["ent_weight"]

    # get task parameters
    is_sim = task_name[:4] == "sim_"
    from constants import SIM_TASK_CONFIGS

    task_config = SIM_TASK_CONFIGS[task_name]
    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]
    rlbench_env = task_config["rlbench_env"]

    # fixed parameters
    state_dim = 8
    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
            "state_dim": state_dim,
            "few_shot": few_shot,
            "ent_weight": ent_weight,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "rlbench_env": rlbench_env,
        "ckpt_names": ckpt_names if ckpt_names else [],
    }
    if is_eval:
        print(f"Evaluating for {ckpt_names}")
        if len(ckpt_names) == 0:
            ckpt_names = [f"policy_best.ckpt"]
        results = []
        for ckpt_name in ckpt_names:
            task_perfs = eval_bc(
                config, ckpt_name, save_episode=True, add_task_ind=add_task_ind
            )
            for key in task_perfs.keys():
                results.append([key, task_perfs[key]])

        for ckpt_name, [success_rate, avg_return] in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        exit()

    required_data_keys = [
        "front_rgb",
        "left_shoulder_rgb",
        "right_shoulder_rgb",
        "wrist_rgb",
        "joint_positions",
        "gripper_open",
    ]
    train_dataloader, val_dataloader = load_rlbench_data(
        dataset_dir=dataset_dir,
        required_data_keys=required_data_keys,
        task_name=task_name,
        chunk_size=args["chunk_size"],
        norm_bound=FRANKA_JOINT_LIMITS,
        batch_size=batch_size,
        add_task_ind=add_task_ind,
        few_shot=few_shot,
    )

    # Save configuration
    config["rlbench_env"] = str(config["rlbench_env"])
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    json_path = os.path.join(ckpt_dir, f"config.json")
    with open(json_path, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(obs, camera_names):
    curr_images = []
    viz_out = {}
    for cam_name in camera_names:
        curr_image = getattr(obs, cam_name)
        viz_out[cam_name] = curr_image
        curr_image = rearrange(curr_image, "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image, viz_out


def eval_bc(config, ckpt_name, save_episode=True, **kwargs):
    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    temporal_agg = config["temporal_agg"]
    rlbench_env = config["rlbench_env"]
    task_name = config["task_name"]

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")

    #  A simple MinMax transformation
    min_bound = FRANKA_JOINT_LIMITS[:, 0]
    max_bound = FRANKA_JOINT_LIMITS[:, 1]
    pre_process = (
        lambda s_pos: (1.0 * (s_pos - min_bound) / (max_bound - min_bound)) * 2.0 - 1
    )
    post_process = (
        lambda s_pos: 1.0 * ((s_pos + 1) / 2) * (max_bound - min_bound) + min_bound
    )

    # Training config
    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks
    num_rollouts = 50
    add_task_ind = kwargs.get("add_task_ind", False)  # check for add_task_ind

    # load simulation environment
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointPosition(), gripper_action_mode=Discrete()
        ),
        obs_config=ObservationConfig(),
        robot_setup="panda",
        headless=not onscreen_render,
    )
    env.launch()

    # Iterate over tasks
    task_performances = {}
    for rlenv in rlbench_env:
        task = env.get_task(rlenv)
        task_dir = os.path.join(
            ckpt_dir, rlenv.__name__, f"{datetime.datetime.now()}"
        )  # Store task specific information here
        if not os.path.isdir(task_dir):
            os.makedirs(task_dir)
        env_max_reward = 1  # Hardcoded as we don't do reward shaping
        episode_returns = []
        highest_rewards = []

        for rollout_id in range(num_rollouts):
            rollout_id += 0
            _, obs = task.reset()
            if temporal_agg:
                all_time_actions = torch.zeros(
                    [max_timesteps, max_timesteps + num_queries, state_dim]
                ).cuda()
            qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
            image_list = []  # for visualization
            qpos_list = []
            target_qpos_list = []
            rewards = []
            skill_ind = torch.tensor([[0, 0]], dtype=torch.float32)
            env_ind = torch.tensor([[0, 0]], dtype=torch.float32)
            if add_task_ind:
                if rlenv in BACKWARD_ENVS:
                    skill_ind = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
                elif rlenv in FORWARD_ENVS:
                    skill_ind = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

                if "box" in rlenv.__name__.lower():
                    env_ind = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
                elif "toilet" in rlenv.__name__.lower():
                    env_ind = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
                elif "grill" in rlenv.__name__.lower():
                    env_ind = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

                # if "box" in rlenv.__name__.lower():
                #     env_ind = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
                # elif "toilet" in rlenv.__name__.lower():
                #     env_ind = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

            skill_ind = skill_ind.cuda()
            env_ind = env_ind.cuda()

            with torch.inference_mode():
                for t in range(max_timesteps):
                    t_skill_ind = skill_ind
                    # if t >= 250:  # change sign of skill_ind for the reverse task
                    #     t_task_ind = skill_ind * -1
                    joint_position = pre_process(obs.joint_positions)
                    qpos_numpy = np.array(np.hstack([joint_position, obs.gripper_open]))
                    qpos = torch.from_numpy(qpos_numpy).float().cuda().unsqueeze(0)
                    qpos_history[:, t] = qpos
                    curr_image, viz_out = get_image(obs, camera_names)
                    image_list.append(viz_out)  # For generating videos

                    ### query policy
                    if config["policy_class"] == "ACT":
                        if t % query_frequency == 0:
                            all_actions = policy(
                                qpos, curr_image, skill_ind=t_skill_ind, env_ind=env_ind
                            )
                        if temporal_agg:
                            all_time_actions[[t], t : t + num_queries] = all_actions
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(
                                actions_for_curr_step != 0, axis=1
                            )
                            actions_for_curr_step = actions_for_curr_step[
                                actions_populated
                            ]
                            k = 0.01
                            exp_weights = np.exp(
                                -k * np.arange(len(actions_for_curr_step))
                            )
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = (
                                torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            )
                            raw_action = (actions_for_curr_step * exp_weights).sum(
                                dim=0, keepdim=True
                            )
                        else:
                            raw_action = all_actions[:, t % query_frequency]
                    elif config["policy_class"] == "CNNMLP":
                        raw_action = policy(qpos, curr_image)
                    else:
                        raise NotImplementedError

                    ### post-process actions
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = np.hstack([post_process(raw_action[:7]), raw_action[-1]])
                    target_qpos = action

                    ### step the environment
                    obs, reward, terminate = task.step(target_qpos)

                    ### for visualization
                    qpos_list.append(qpos_numpy)
                    target_qpos_list.append(target_qpos)
                    rewards.append(reward)

            rewards = np.array(rewards)
            episode_return = np.sum(rewards[rewards != None])
            episode_returns.append(episode_return)
            episode_highest_reward = np.max(rewards)
            highest_rewards.append(episode_highest_reward)
            print(
                f"Rollout for {rlenv}: {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
            )
            if save_episode:
                save_videos(
                    image_list,
                    video_path=os.path.join(
                        task_dir,
                        f"video{rollout_id}_{episode_highest_reward==env_max_reward}.mp4",
                    ),
                )

        success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
        avg_return = np.mean(episode_returns)
        summary_str = f"Task: {rlenv}\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
        for r in range(env_max_reward + 1):
            more_or_equal_r = (np.array(highest_rewards) >= r).sum()
            more_or_equal_r_rate = more_or_equal_r / num_rollouts
            summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

        print(summary_str)

        # save success rate to txt
        result_file_name = "result_" + ckpt_name.split(".")[0] + f".txt"
        with open(os.path.join(task_dir, result_file_name), "w") as f:
            f.write(str(config))
            f.write(summary_str)
            f.write(repr(episode_returns))
            f.write("\n\n")
            f.write(repr(highest_rewards))

        task_performances[rlenv.__name__] = [success_rate, avg_return]
    env.shutdown()
    return task_performances


def forward_pass(data, policy):
    images = data["images"]
    joint_action = data["joint_action"]
    is_pad = data["is_pad"]
    gripper_action = data["gripper_action"]
    skill_ind = data["skill_ind"]
    env_ind = data["env_ind"]

    # Append gripper action to joint action
    action = torch.concatenate((joint_action, gripper_action.unsqueeze(-1)), 2)

    # Get the current joint state
    qpos = action[:, 0, :]
    # qpos += torch.normal(0, 0.01, size=qpos.shape)

    images, qpos, action, is_pad, skill_ind, env_ind = (
        images.cuda(),
        qpos.cuda(),
        action.cuda(),
        is_pad.cuda(),
        skill_ind.cuda(),
        env_ind.cuda(),
    )
    return policy(
        qpos,
        images,
        actions=action,
        is_pad=is_pad,
        skill_ind=skill_ind,
        env_ind=env_ind,
    )  # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    ckpt_names = config.get("ckpt_names", [])

    set_seed(seed)
    policy = make_policy(policy_class, policy_config)
    optimizer = make_optimizer(policy_class, policy)

    if len(ckpt_names) > 0:
        # load policy and stats
        ckpt_path = os.path.join(ckpt_names[0])
        loading_status = policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        print(f"******** Loaded model: {ckpt_path} ********")
    else:
        print("******** Training the model from scratch ********")

    policy.cuda()
    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch {epoch}")
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * epoch : (batch_idx + 1) * (epoch + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(
        f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}"
    )

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument(
        "--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True
    )
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument(
        "--task_name", action="store", type=str, help="task_name", required=True
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, help="batch_size", required=True
    )
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument(
        "--num_epochs", action="store", type=int, help="num_epochs", required=True
    )
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    # for ACT
    parser.add_argument(
        "--kl_weight", action="store", type=int, help="KL Weight", required=False
    )
    parser.add_argument(
        "--chunk_size", action="store", type=int, help="chunk_size", required=False
    )
    parser.add_argument(
        "--hidden_dim", action="store", type=int, help="hidden_dim", required=False
    )
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--add_task_ind", action="store_true")

    parser.add_argument("--ckpt_names", action="store", nargs="*", help="ckpt_names")
    parser.add_argument("--few_shot", action="store", type=int, help="few_shot")
    parser.add_argument("--ent_weight", action="store", type=float, help="ent_weight")

    main(vars(parser.parse_args()))
