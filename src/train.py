import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
import math

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb
import pdb

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func,
               mode="decoder",
               lasso_guided_opt_lam=None,
               lasso_guided_opt_layer=-2,
               lasso_guided_opt_token=-1,
               w_star=None):
    optimizer.zero_grad()
    if mode == "encoder" and lasso_guided_opt_lam is not None:
        output, hidden_states = model(xs, ys, return_hidden_states=True)
    else:
        output = model(xs, ys)
    if mode == "decoder":
        loss = loss_func(output, ys)
    elif mode == "encoder":
        # Predict on final token only in encoder mode
        loss = loss_func(output[:, -1:], ys[:, -1:])
        if lasso_guided_opt_lam is not None:
            B, N, d = xs.shape
            # compute loss between second-to-last layer, and true w_star which has shape Bxdx1
            w_star = w_star.to(xs.device)
            Bw, dw, _ = w_star.shape
            assert Bw == B and dw == d
            w_star = w_star.squeeze(2).view([B, 1, d])
            loss += lasso_guided_opt_lam * ((hidden_states[lasso_guided_opt_layer][:, :lasso_guided_opt_token, -d:] - w_star)**2).sum(dim=2).mean()
    else:
        raise NotImplementedError
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        if not args.training.optimizer_reset:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

        # refresh learning rate to the one specified by training args
        if args.training.learning_rate_override:
            for g in optimizer.param_groups:
                g['lr'] = args.training.learning_rate


    all_task_names = [t.name for t in args.training.tasks]

    n_dims = args.model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)

    all_task_samplers = [get_task_sampler(
        task.name,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **task.kwargs,
    ) for task in args.training.tasks]
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:

        # Loop over all tasks
        for (task_name, task_sampler) in zip(all_task_names, all_task_samplers):

            data_sampler_args = {}
            task_sampler_args = {}

            if "sparse" in task_name:
                task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
            if num_training_examples is not None:
                assert num_training_examples >= bsize
                seeds = sample_seeds(num_training_examples, bsize)
                data_sampler_args["seeds"] = seeds
                task_sampler_args["seeds"] = [s + 1 for s in seeds]

            xs = data_sampler.sample_xs(
                curriculum.n_points,
                bsize,
                curriculum.n_dims_truncated,
                **data_sampler_args,
            )
            task = task_sampler(**task_sampler_args)
            ys = task.evaluate(xs)

            loss_func = task.get_training_metric()

            encoder_decoder_mode = "encoder" if args.model.family == "EncoderTF" else "decoder"

            if task_name == "sparse_linear_regression" and args.training.lasso_guided_opt:
                w_b = task.w_b
                if task.normalize_w:
                    w_b = w_b * task.scale / math.sqrt(task.sparsity)

                loss, output = train_step(
                    model, xs.cuda(), ys.cuda(), optimizer, loss_func,
                    mode=encoder_decoder_mode,
                    lasso_guided_opt_lam=args.training.lasso_guided_opt_lam,
                    lasso_guided_opt_layer=args.training.lasso_guided_opt_layer,
                    lasso_guided_opt_token=args.training.lasso_guided_opt_token,
                    w_star=w_b,
                )
            else:
                loss, output = train_step(
                    model, xs.cuda(), ys.cuda(), optimizer, loss_func,
                    mode=encoder_decoder_mode,
                )

            point_wise_tags = list(range(curriculum.n_points))
            point_wise_loss_func = task.get_metric()
            point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

            baseline_loss = (
                sum(
                    max(curriculum.n_dims_truncated - ii, 0)
                    for ii in range(curriculum.n_points)
                )
                / curriculum.n_points
            )

            if i % args.wandb.log_every_steps == 0 and not args.test_run:
                wandb.log(
                    {
                        f"{task_name}/overall_loss": loss,
                        f"{task_name}/excess_loss": loss / baseline_loss,
                        f"{task_name}/pointwise/loss": dict(
                            zip(point_wise_tags, point_wise_loss.cpu().numpy())
                        ),
                        "n_points": curriculum.n_points,
                        "n_dims": curriculum.n_dims_truncated,
                    },
                    step=i,
                )

            pbar.set_description(f"loss {loss}")

        # TASK FOR LOOP ENDS

        curriculum.update()

        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 1000
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm", "EncoderTF"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
