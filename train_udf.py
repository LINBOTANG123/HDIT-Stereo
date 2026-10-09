#!/usr/bin/env python3

"""Trains Karras et al. (2022) diffusion models."""

import argparse
from copy import deepcopy
from functools import partial
import importlib.util
import sys
import math
import json
from pathlib import Path
import time

import accelerate
import safetensors.torch as safetorch
import torch
import torch._dynamo
import torch.nn.functional as F
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import optim
from torch.utils import data, flop_counter
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm
import pdb
import numpy as np

import k_diffusion as K


def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=64,
                   help='the batch size')
    p.add_argument('--checkpointing', action='store_true',
                   help='enable gradient checkpointing')
    p.add_argument('--compile', action='store_true',
                   help='compile the model')
    p.add_argument('--config', type=str, required=True,
                   help='the configuration file')
    p.add_argument('--demo-every', type=int, default=500,
                   help='save a demo grid every this many steps')
    p.add_argument('--end-step', type=int, default=None,
                   help='the step to end training at')
    p.add_argument('--evaluate-every', type=int, default=10000,
                   help='evaluate every this many steps')
    p.add_argument('--evaluate-only', action='store_true',
                   help='evaluate instead of training')
    p.add_argument('--gns', action='store_true',
                   help='measure the gradient noise scale (DDP only, disables stratified sampling)')
    p.add_argument('--grad-accum-steps', type=int, default=1,
                   help='the number of gradient accumulation steps')
    p.add_argument('--lr', type=float,
                   help='the learning rate')
    p.add_argument('--mixed-precision', type=str,
                   help='the mixed precision type')
    p.add_argument('--name', type=str, default='model',
                   help='the name of the run')
    p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
    p.add_argument('--reset-ema', action='store_true',
                   help='reset the EMA')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    p.add_argument('--resume-inference', type=str,
                   help='the inference checkpoint to resume from')
    p.add_argument('--sample-n', type=int, default=64,
                   help='the number of images to sample for demo grids')
    p.add_argument('--save-every', type=int, default=10000,
                   help='save every this many steps')
    p.add_argument('--seed', type=int,
                   help='the random seed')
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    p.add_argument('--wandb-entity', type=str,
                   help='the wandb entity name')
    p.add_argument('--wandb-group', type=str,
                   help='the wandb group name')
    p.add_argument('--wandb-project', type=str,
                   help='the wandb project name (specify this to enable wandb)')
    p.add_argument('--wandb-save-model', action='store_true',
                   help='save model to wandb')
    
    p.add_argument('--bw-enable', action='store_true',
               help='Enable boundary-weighted auxiliary loss on the zero-level band.')
    p.add_argument('--bw-tau-px', type=float, default=2.0,
                help='Half-width of boundary band τ (in pixels).')
    p.add_argument('--bw-alpha', type=float, default=5.0,
                help='Band emphasis α (1 + α inside the band).')
    p.add_argument('--bw-lambda', type=float, default=0.3,
                help='Mixing weight λ for the boundary loss term.')
    p.add_argument('--bw-soft-beta-px', type=float, default=0.0,
                help='If >0, use soft Gaussian band with β (pixels); if 0, use hard band.')
    p.add_argument('--udf-k', type=float, default=3.0,
                help='Steepness k for UDF nonlinearity applied before loss: 1 - exp(-k * udf_norm).')
    p.add_argument('--disp-norm', type=float, default=None,
                help='Disparity normalization divisor (pixels). Defaults to model config disp_norm.')
    p.add_argument('--udf-channel-weight', type=float, default=1.0,
                help='Scale UDF channel by this factor in transform space, boosting its MSE contribution by factor^2.')
    p.add_argument('--fg-alpha', type=float, default=0.0,
                help='Foreground weight multiplier: pixels with disp>0 get weight (1+fg_alpha). 0=disabled.')
    p.add_argument('--cross-view-diff', action='store_true',
                help='Add tL-tR cross-view difference feature to conditioning. '
                     'Encodes stereo ambiguity signal (near 0 for no-texture camouflage).')
    p.add_argument('--refine-head', action='store_true',
                help='Append a lightweight conv refinement head after patch_out. '
                     'Uses stereo images as guidance to resolve within-patch disparity gradients.')
    p.add_argument('--use-scc', action='store_true',
                help='Enable Lean Correlation Conditioner: a symmetric (cyclopean) cost-volume '
                     'joint structure from L/R injected into the token stream.')
    p.add_argument('--scc-disp-weight', type=float, default=0.0,
                help='Weight for SCC disparity supervision (soft-argmax vs GT disp). 0=disabled.')
    p.add_argument('--scc-dmax', type=int, default=16,
                help='Max disparity (in tokens) searched by the SCC cost volume.')
    p.add_argument('--scc-disp-step', type=float, default=1.0,
                help='Disparity-candidate step in tokens for the SCC cost volume. '
                     '1.0=integer tokens (px_per_token px); 0.5=half-token (finer/sub-pixel).')
    p.add_argument('--scc-plane', action='store_true',
                help='A1+B1 LCC upgrade: slant head predicts (dx, dy) disparity gradients per '
                     'token and the full cost-volume distribution P is injected into fuse '
                     '(not just collapsed d_hat). Adds a gradient supervision term weighted by '
                     '--scc-disp-weight. Changes fuse input channels, needs fresh checkpoint.')
    p.add_argument('--scc-nowrap', action='store_true',
                help='Skip the cyclopean warp and the fuse network entirely: project the fused '
                     'cost-volume distribution P (and slant, if --scc-plane) directly to width0 '
                     'with a single conv, then inject. Composable with --scc-plane. '
                     'Different module weights than the default warp+fuse path, needs fresh checkpoint.')
    p.add_argument('--disp-left', action='store_true',
                help='Left-view matching instead of cyclopean: fL stays on its native grid, only '
                     'fR is sampled at x - d, so the LCC disparity estimate lives directly on the '
                     'left-image pixel grid (matching GT disparity convention and the main model\'s '
                     'own output channel). Composable with --scc-plane / --scc-nowrap. '
                     'Same tensor shapes as cyclopean but different learned semantics, needs fresh checkpoint.')
    p.add_argument('--use-scc-native', action='store_true',
                help='Native-resolution channel-concat LCC: stride-1 encoder produces d_hat at '
                     'full image resolution (no upsampling). Changes in_channels, needs fresh checkpoint.')
    p.add_argument('--scc-native-dmax', type=int, default=64,
                help='Max disparity in pixels for the native-resolution LCC.')
    p.add_argument('--scc-native-feat-ch', type=int, default=32,
                help='Encoder channel count for the native-resolution LCC.')
    p.add_argument('--scc-native-disp-step', type=float, default=4.0,
                help='Pixel step between disparity candidates for the native LCC (4=same granularity as token-level).')
    p.add_argument('--scc-dino-only', action='store_true',
                help='DINO-cost-volume-only mode: agg/slant read only the DINO cost volume, '
                     'skipping LCC\'s own CNN correlation channels entirely (still requires '
                     '--use-scc and use_scc_dino_cv=True in the config -- LCC\'s encoder still '
                     'runs for the appearance warp+fuse step, and the soft-argmax output stays '
                     'on LCC\'s own token-grid scale). Not supported with use_scc_c2f. '
                     'Changes agg/slant input channels, needs fresh checkpoint.')
    p.add_argument('--use-raft-disp', action='store_true',
                help='Channel-concat external sharp disparity before patch_in. '
                     'Uses GT disparity (+ noise) during training; expects raft_disp kwarg at inference.')
    p.add_argument('--raft-disp-noise-std', type=float, default=1.0,
                help='Std of Gaussian noise (px) added to GT disp proxy at train time to simulate RAFT errors.')
    p.add_argument('--raft-disp-dropout', type=float, default=0.1,
                help='Probability of zeroing the disp channel per sample (trains robustness to missing input).')
    p.add_argument('--amodal-loss-weight', type=float, default=0.0,
                help='Weight for auxiliary amodal mask BCE loss. 0=disabled (original behavior).')
    p.add_argument('--amodal-intersection-boost', type=float, default=0.0,
                help='Extra BCE weight multiplier for pixels in the obj1∩obj2 intersection region. '
                     '0=uniform weighting. e.g. 9.0 means intersection pixels get 10× weight.')
    p.add_argument('--deterministic', action='store_true',
                   help='Train the SAME network as a deterministic regressor instead of a diffusion '
                        'model: input is all zeros at a fixed sigma, the net regresses x0 directly '
                        '(plain MSE, no noise). Same arch/conditioning/channel-scales/scc losses. '
                        'Recorded in the checkpoint config so infer.py does one forward pass.')
    p.add_argument('--deterministic-sigma', type=float, default=1.0,
                   help='Fixed noise-level conditioning value fed to the net in --deterministic mode.')
    p.add_argument('--loss-mode', type=str, default='joint',
               choices=['joint', 'udf_only', 'disp_only'],
               help='Which channel loss to optimize.')
    p.add_argument(
        "--disable-udf-boundary-focus",
        action="store_true",
        help="Disable exponential UDF boundary-focus transform and use raw normalized UDF."
    )
    p.add_argument(
        "--two-object-only",
        action="store_true",
        help="Skip one_object_* samples during training (train and val). "
             "Useful for layered UDF models where one-object data adds no signal for udf_obj2."
    )

    # ── Distillation (fully opt-in: no effect on any existing script unless
    #    --teacher-checkpoint is set) ──────────────────────────────────────
    p.add_argument('--teacher-checkpoint', type=str, default=None,
                help='Path to a teacher .pth checkpoint for distillation. When set, a frozen '
                     'teacher model (built from --teacher-config, which may be a different '
                     'size than the student defined by --config) is loaded, and its predicted '
                     'output at the same noised input/timestep/conditioning is blended into '
                     'the training loss via --distill-weight. When unset (default), this '
                     'entire code path is skipped -- training is byte-for-byte unaffected.')
    p.add_argument('--teacher-config', type=str, default=None,
                help='Config JSON used to build the teacher architecture. Required if '
                     '--teacher-checkpoint is set.')
    p.add_argument('--distill-weight', type=float, default=0.5,
                help='Blend weight in [0,1] between the teacher-matching loss and the '
                     'ground-truth loss: loss = (1-w)*ground_truth + w*teacher_match. '
                     '0=pure ground truth (teacher unused in the loss), 1=pure distillation. '
                     'Only used if --teacher-checkpoint is set.')
    p.add_argument('--distill-use-ema', dest='distill_use_ema', action='store_true', default=True,
                help="Use the teacher checkpoint's EMA weights as the teacher (default: on).")
    p.add_argument('--distill-use-raw', dest='distill_use_ema', action='store_false',
                help="Use the teacher checkpoint's raw (non-EMA) weights instead of EMA.")

    args = p.parse_args()

    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass

    config = K.config.load_config(args.config)
    model_config = config['model']
    if args.deterministic:
        config['deterministic'] = {'sigma': args.deterministic_sigma}
    det_cfg = config.get('deterministic')
    dataset_config = config['dataset']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_sched_config = config['ema_sched']

    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps, mixed_precision=args.mixed_precision)
    ensure_distributed()
    device = accelerator.device
    unwrap = accelerator.unwrap_model
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'World size: {accelerator.num_processes}', flush=True)
        print(f'Batch size: {args.batch_size * accelerator.num_processes}', flush=True)

    if args.seed is not None:
        seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
        torch.manual_seed(seeds[accelerator.process_index])
    demo_gen = torch.Generator().manual_seed(torch.randint(-2 ** 63, 2 ** 63 - 1, ()).item())
    elapsed = 0.0

    if args.cross_view_diff:
        model_config['use_cross_view_diff'] = True
    if args.refine_head:
        model_config['use_refine_head'] = True
    if args.use_scc:
        model_config['use_scc'] = True
        model_config['scc_dmax'] = args.scc_dmax
        model_config['scc_disp_step'] = args.scc_disp_step
        if args.scc_plane:
            model_config['scc_plane'] = True
        if args.scc_nowrap:
            model_config['scc_nowrap'] = True
        if args.disp_left:
            model_config['disp_left'] = True
        if args.scc_dino_only:
            model_config['scc_use_own_cv'] = False
    if args.use_scc_native:
        model_config['use_scc_native'] = True
        model_config['scc_native_dmax'] = args.scc_native_dmax
        model_config['scc_native_feat_ch'] = args.scc_native_feat_ch
        model_config['scc_native_disp_step'] = args.scc_native_disp_step
    if args.use_raft_disp:
        model_config['use_raft_disp'] = True

    inner_model = K.config.make_model(config)
    inner_model_ema = deepcopy(inner_model)

    if args.compile:
        inner_model.compile()
        # inner_model_ema.compile()

    if accelerator.is_main_process:
        print(f'Parameters: {K.utils.n_params(inner_model):,}')

    lr = opt_config['lr'] if args.lr is None else args.lr
    groups = inner_model.param_groups(lr)
    if opt_config['type'] == 'adamw':
        opt = optim.AdamW(groups,
                          lr=lr,
                          betas=tuple(opt_config['betas']),
                          eps=opt_config['eps'],
                          weight_decay=opt_config['weight_decay'])
    elif opt_config['type'] == 'adam8bit':
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(groups,
                                 lr=lr,
                                 betas=tuple(opt_config['betas']),
                                 eps=opt_config['eps'],
                                 weight_decay=opt_config['weight_decay'])
    elif opt_config['type'] == 'sgd':
        opt = optim.SGD(groups,
                        lr=lr,
                        momentum=opt_config.get('momentum', 0.),
                        nesterov=opt_config.get('nesterov', False),
                        weight_decay=opt_config.get('weight_decay', 0.))
    else:
        raise ValueError('Invalid optimizer type')

    if sched_config['type'] == 'inverse':
        sched = K.utils.InverseLR(opt,
                                  inv_gamma=sched_config['inv_gamma'],
                                  power=sched_config['power'],
                                  warmup=sched_config['warmup'])
    elif sched_config['type'] == 'exponential':
        sched = K.utils.ExponentialLR(opt,
                                      num_steps=sched_config['num_steps'],
                                      decay=sched_config['decay'],
                                      warmup=sched_config['warmup'])
    elif sched_config['type'] == 'constant':
        sched = K.utils.ConstantLRWithWarmup(opt, warmup=sched_config['warmup'])
    else:
        raise ValueError('Invalid schedule type')

    assert ema_sched_config['type'] == 'inverse'
    ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                  max_value=ema_sched_config['max_value'])
    ema_stats = {}

    tf = transforms.Compose([
        transforms.Resize(size[0], interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size[0]),
        K.augmentation.KarrasAugmentationPipeline(model_config['augment_prob'], disable_all=model_config['augment_prob'] == 0),
        # transforms.ToTensor(),
    ])

    if dataset_config['type'] == 'imagefolder':
        train_set = K.utils.FolderOfImages(dataset_config['location'], transform=tf)
    elif dataset_config['type'] == 'imagefolder-class':
        train_set = datasets.ImageFolder(dataset_config['location'], transform=tf)
    elif dataset_config['type'] == 'cifar10':
        train_set = datasets.CIFAR10(dataset_config['location'], train=True, download=True, transform=tf)
    elif dataset_config['type'] == 'mnist':
        train_set = datasets.MNIST(dataset_config['location'], train=True, download=True, transform=tf)
    elif dataset_config['type'] == 'huggingface':
        from datasets import load_dataset
        train_set = load_dataset(dataset_config['location'])
        train_set.set_transform(partial(K.utils.hf_datasets_augs_helper, transform=tf, image_key=dataset_config['image_key']))
        train_set = train_set['train']
    elif dataset_config['type'] == 'custom':
        location = (Path(args.config).parent / dataset_config['location']).resolve()
        parent_dir = str(location.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        module_name = location.stem
        spec = importlib.util.spec_from_file_location(module_name, location)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        get_dataset = getattr(module, dataset_config.get('get_dataset', 'FoJDataset'))
        custom_dataset_config = dataset_config.get('config', {})
        if args.two_object_only:
            custom_dataset_config = dict(custom_dataset_config, two_object_only=True)

        # 1) Instantiate without transform arg
        train_set = get_dataset(**custom_dataset_config)

        # after creating `train_set` and `get_dataset` above
        val_cfg = dataset_config.get("val_config", None)
        if val_cfg is not None:
            if args.two_object_only:
                val_cfg = dict(val_cfg, two_object_only=True)
            val_set = get_dataset(**val_cfg)
        else:
            # fallback: small random split from train_set if no val_config provided
            n_total = len(train_set)
            n_val   = max(1, int(0.05 * n_total))
            n_train = n_total - n_val
            train_set, val_set = torch.utils.data.random_split(
                train_set, [n_train, n_val],
                generator=torch.Generator().manual_seed(123)
            )

        val_dl = data.DataLoader(
            val_set,
            args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    # If logging to wandb, initialize the run
    use_wandb = accelerator.is_main_process and args.wandb_project
    if use_wandb:
        import wandb
        log_config = vars(args)
        log_config['config'] = config
        log_config['parameters'] = K.utils.n_params(inner_model)
        wandb.init(project=args.wandb_project,
            settings=wandb.Settings(_service_wait=300),
            entity=args.wandb_entity, group=args.wandb_group, config=log_config, save_code=True)
        
    if accelerator.is_main_process:
        try:
            print(f'Number of items in dataset: {len(train_set):,}')
        except TypeError:
            pass

    image_key = dataset_config.get('image_key', 0)
    num_classes = dataset_config.get('num_classes', 0)
    cond_dropout_rate = dataset_config.get('cond_dropout_rate', 0.1)
    class_key = dataset_config.get('class_key', 1)

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    # inner_model, inner_model_ema, opt, train_dl = accelerator.prepare(inner_model, inner_model_ema, opt, train_dl)
    inner_model, inner_model_ema, opt, train_dl, val_dl = accelerator.prepare(inner_model, inner_model_ema, opt, train_dl, val_dl)


    with torch.no_grad(), K.models.flops.flop_counter() as fc:
        x     = torch.zeros([1, model_config['input_channels'], *size],
                            device=device)
        sigma = torch.ones([1], device=device)
        extra = {}

        if getattr(unwrap(inner_model), "num_classes", 0):
            extra["class_cond"] = torch.zeros([1], dtype=torch.long,
                                            device=device)

        # ---------- choose the right dummy aug_cond ----------
        m = unwrap(inner_model)
        needs_image_cond = (
            getattr(m, "expects_image_aug_cond", False) or
            hasattr(m, "image_encoder") or                  # old version
            hasattr(m, "image_encoder_global") or           # new version
            (hasattr(m, "mapping_cond_in_proj") and m.mapping_cond_in_proj is not None)
        )
        if needs_image_cond:
            cond_ch = model_config.get("cond_channels", 3)
            # use random (or any non-zero) dummy to avoid confusing prints/asserts
            extra["aug_cond"] = torch.randn([1, cond_ch, *size], device=device)
        else:
            # vanilla 9-dim augmentation vector path
            extra["aug_cond"] = torch.zeros([1, 9], device=device)
        # ------------------------------------------------------
        # use_scc_c2f requires a wide L/R crop -- this dummy FLOP-counting
        # pass doesn't go through the dataset, so it never gets one from a
        # real batch; without this the very first forward call at startup
        # crashes before training even begins (LeanCorrelationConditionerC2F
        # raises if scc_coarse_L/R are missing). Width matches whatever the
        # dataset config actually uses, so the FLOP count is representative,
        # not just non-crashing.
        if getattr(m, "use_scc_c2f", False):
            coarse_strip_px = dataset_config.get('config', {}).get('coarse_strip_px', 512)
            extra["scc_coarse_L"] = torch.randn([1, 3, size[0], coarse_strip_px], device=device)
            extra["scc_coarse_R"] = torch.randn([1, 3, size[0], coarse_strip_px], device=device)
        # ------------------------------------------------------

        inner_model(x, sigma, **extra)
        if accelerator.is_main_process:
            print(f"Forward pass GFLOPs: {fc.flops/1e9:,.3f}")


    if use_wandb:
        wandb.watch(inner_model)
    if accelerator.num_processes == 1:
        args.gns = False
    if args.gns:
        gns_stats_hook = K.gns.DDPGradientStatsHook(inner_model)
        gns_stats = K.gns.GradientNoiseScale()
    else:
        gns_stats = None
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    sample_density = K.config.make_sample_density(model_config)

    model = K.config.make_denoiser_wrapper(config)(inner_model)
    model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema)

    # ── Optional distillation teacher (opt-in; None unless --teacher-checkpoint is set) ──
    teacher_model = None
    if args.teacher_checkpoint:
        if not args.teacher_config:
            raise ValueError('--teacher-checkpoint requires --teacher-config (the teacher can '
                              'be a different architecture size than the student).')
        if accelerator.is_main_process:
            print(f'Loading distillation teacher from {args.teacher_checkpoint} '
                  f'(arch: {args.teacher_config}, '
                  f'weights: {"model_ema" if args.distill_use_ema else "model"})...')
        teacher_config = K.config.load_config(args.teacher_config)
        teacher_inner_model = K.config.make_model(teacher_config)
        teacher_ckpt = torch.load(args.teacher_checkpoint, map_location='cpu')
        teacher_state_key = 'model_ema' if args.distill_use_ema else 'model'
        teacher_inner_model.load_state_dict(teacher_ckpt[teacher_state_key])
        del teacher_ckpt
        for p_ in teacher_inner_model.parameters():
            p_.requires_grad_(False)
        teacher_inner_model.eval()
        teacher_inner_model.to(device)
        teacher_model = K.config.make_denoiser_wrapper(teacher_config)(teacher_inner_model)
        if accelerator.is_main_process:
            print(f'Teacher parameters (frozen): {K.utils.n_params(teacher_inner_model):,}')

    state_path = Path(f'{args.name}_state.json')

    if state_path.exists() or args.resume:
        if args.resume:
            ckpt_path = args.resume
        if not args.resume:
            state = json.load(open(state_path))
            ckpt_path = state['latest_checkpoint']
        if accelerator.is_main_process:
            print(f'Resuming from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        unwrap(model.inner_model).load_state_dict(ckpt['model'])
        unwrap(model_ema.inner_model).load_state_dict(ckpt['model_ema'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        ema_sched.load_state_dict(ckpt['ema_sched'])
        ema_stats = ckpt.get('ema_stats', ema_stats)
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        if args.gns and ckpt.get('gns_stats', None) is not None:
            gns_stats.load_state_dict(ckpt['gns_stats'])
        demo_gen.set_state(ckpt['demo_gen'])
        elapsed = ckpt.get('elapsed', 0.0)

        del ckpt
    else:
        epoch = 0
        step = 0

    if args.reset_ema:
        unwrap(model.inner_model).load_state_dict(unwrap(model_ema.inner_model).state_dict())
        ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                      max_value=ema_sched_config['max_value'])
        ema_stats = {}

    if args.resume_inference:
        if accelerator.is_main_process:
            print(f'Loading {args.resume_inference}...')
        ckpt = safetorch.load_file(args.resume_inference)
        unwrap(model.inner_model).load_state_dict(ckpt)
        unwrap(model_ema.inner_model).load_state_dict(ckpt)
        del ckpt

    evaluate_enabled = (args.evaluate_every > 0) and (val_dl is not None)

    cfg_scale = 1.

    def make_cfg_model_fn(model):
        def cfg_model_fn(x, sigma, class_cond):
            x_in = torch.cat([x, x])
            sigma_in = torch.cat([sigma, sigma])
            class_uncond = torch.full_like(class_cond, num_classes)
            class_cond_in = torch.cat([class_uncond, class_cond])
            out = model(x_in, sigma_in, class_cond=class_cond_in)
            out_uncond, out_cond = out.chunk(2)
            return out_uncond + (out_cond - out_uncond) * cfg_scale
        if cfg_scale != 1:
            return cfg_model_fn
        return model

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def demo():
        if accelerator.is_main_process:
            tqdm.write("Sampling…")

        filename = f"{args.name}_demo_{step:08}.png"

        # ───────────────────────────────────────────────────────────
        # 1. Allocate latent noise  x  exactly as before
        # ───────────────────────────────────────────────────────────
        n_per_proc = math.ceil(args.sample_n / accelerator.num_processes)

        x = torch.randn(
                [accelerator.num_processes, n_per_proc,
                model_config["input_channels"], size[0], size[1]],
                generator=demo_gen
            ).to(device)                                    # ← on GPU
        dist.broadcast(x, 0)
        x = x[accelerator.process_index] * sigma_max

        # ───────────────────────────────────────────────────────────
        # 2. Build   extra_args   for the FoJ model
        #    • we still support class_cond, cfg, etc.
        #    • BUT we *add* an RGB conditioning image tensor
        # ───────────────────────────────────────────────────────────
        model_fn, extra_args = model_ema, {}

        # 2-a class-conditioning (unchanged)
        if num_classes:
            class_cond = torch.randint(
                0, num_classes,
                [accelerator.num_processes, n_per_proc],
                generator=demo_gen, device=device)
            dist.broadcast(class_cond, 0)
            extra_args["class_cond"] = class_cond[accelerator.process_index]
            model_fn = make_cfg_model_fn(model_ema)

        def extract_cond(sample):
            """Returns the cond tensor (index 2) from a dataset element: (reals, aug_vec, cond, ...)."""
            if isinstance(sample, (tuple, list)) and len(sample) == 2 and torch.is_tensor(sample[1]):
                sample = sample[0]
            return sample[2]  # cond6: (6,H,W) for stereo

        def extract_coarse(sample):
            """Returns (scc_coarse_L, scc_coarse_R) from a dataset element, honoring
            the same optional-field ordering as the training loop's _next_idx logic."""
            if isinstance(sample, (tuple, list)) and len(sample) == 2 and torch.is_tensor(sample[1]):
                sample = sample[0]
            idx = 4
            if args.amodal_loss_weight > 0.0 and has_amodal_masks:
                idx += 1
            if has_disp_valid:
                idx += 1
            return sample[idx], sample[idx + 1]


        cond_channels = model_config.get("cond_channels", 6)
        if accelerator.is_main_process:
            gen  = torch.Generator().manual_seed(step)
            idxs = torch.randint(0, len(train_set),
                                (n_per_proc,), generator=gen).tolist()

            cond_rank0 = torch.stack([extract_cond(train_set[i]) for i in idxs])
                                                            # [n, cond_channels, H, W]
        else:
            cond_rank0 = torch.empty(
                [n_per_proc, cond_channels, *size], dtype=torch.float32, device=device)

        # move to GPU + dtype fix + normalise
        cond_rank0 = cond_rank0.to(device, dtype=torch.float32)
        if cond_rank0.max() > 1:                # dataset might be 0-255
            cond_rank0.div_(255.)

        dist.broadcast(cond_rank0, 0)

        extra_args = {"aug_cond": cond_rank0}

        if model_wants_c2f:
            if not has_coarse_strip:
                raise ValueError(
                    "Model has use_scc_c2f=True but the dataset wasn't built with "
                    "return_coarse_strip=True -- set it in the dataset config.")
            coarse_strip_px = dataset_config.get('config', {}).get('coarse_strip_px', 512)
            if accelerator.is_main_process:
                coarse_pairs = [extract_coarse(train_set[i]) for i in idxs]
                scc_coarse_L_rank0 = torch.stack([p[0] for p in coarse_pairs])
                scc_coarse_R_rank0 = torch.stack([p[1] for p in coarse_pairs])
            else:
                scc_coarse_L_rank0 = torch.empty(
                    [n_per_proc, 3, size[0], coarse_strip_px], dtype=torch.float32, device=device)
                scc_coarse_R_rank0 = torch.empty(
                    [n_per_proc, 3, size[0], coarse_strip_px], dtype=torch.float32, device=device)
            scc_coarse_L_rank0 = scc_coarse_L_rank0.to(device, dtype=torch.float32)
            scc_coarse_R_rank0 = scc_coarse_R_rank0.to(device, dtype=torch.float32)
            dist.broadcast(scc_coarse_L_rank0, 0)
            dist.broadcast(scc_coarse_R_rank0, 0)
            extra_args["scc_coarse_L"] = scc_coarse_L_rank0
            extra_args["scc_coarse_R"] = scc_coarse_R_rank0

        # ───────────────────────────────────────────────────────────
        # 3. Sample with K-Diffusion’s DPMPP-2M-SDE solver
        # ───────────────────────────────────────────────────────────
        sigmas = K.sampling.get_sigmas_karras(
                    50, sigma_min, sigma_max, rho=7., device=device)

        if det_cfg:
            x_0 = model_fn.inner_model(
                torch.zeros_like(x), x.new_full([x.shape[0]], det_cfg['sigma']), **extra_args)
        else:
            x_0 = K.sampling.sample_dpmpp_2m_sde(
                model_fn, x, sigmas,
                extra_args=extra_args,
                eta=0.0, solver_type="heun",
                disable=not accelerator.is_main_process)

        # gather tensors to rank-0
        x_0       = accelerator.gather(x_0)[:args.sample_n]      # (N,7,H,W)
        cond_imgs = accelerator.gather(cond_rank0)[:args.sample_n]  # (N,3,H,W)

        if accelerator.is_main_process:
            import numpy as np
            from PIL import Image
            import os

            out_dir = f"{args.name}_demo"               # e.g. foj_diffusion_memfit_demo
            os.makedirs(out_dir, exist_ok=True)

            H, W = x_0.shape[2:]

            inv = inverse_transform_pred(x_0)
            is_3ch_demo = (len(inv) == 3)
            if is_3ch_demo:
                udf1_px, udf2_px, disp_px = inv
            else:
                udf1_px, disp_px = inv
                udf2_px = None

            for k in range(x_0.size(0)):
                # 1. save predicted UDF and disparity in pixel units
                if udf1_px is not None:
                    np.save(os.path.join(
                        out_dir, f"{args.name}_step{step:08}_sample{k:02}_udf_obj1.npy"),
                        udf1_px[k, 0].cpu().numpy().astype(np.float32))
                if udf2_px is not None:
                    np.save(os.path.join(
                        out_dir, f"{args.name}_step{step:08}_sample{k:02}_udf_obj2.npy"),
                        udf2_px[k, 0].cpu().numpy().astype(np.float32))
                np.save(os.path.join(
                    out_dir, f"{args.name}_step{step:08}_sample{k:02}_disp.npy"),
                    disp_px[k, 0].cpu().numpy().astype(np.float32))

                # 2. save conditioning RGB images — left (ch 0:3) and right (ch 3:6)
                cond = cond_imgs[k].cpu().clamp(0, 1)
                for side, ch in (("left", slice(0, 3)), ("right", slice(3, 6))):
                    side_np = (cond[ch].permute(1, 2, 0) * 255).byte().numpy()
                    Image.fromarray(side_np).save(os.path.join(
                        out_dir, f"{args.name}_step{step:08}_sample{k:02}_{side}.png"))
                
    u_scale = None
    try:
        u_scale = config["dataset"]["val_config"].get("u_scale", None)
    except Exception:
        u_scale = config["dataset"].get("config", {}).get("u_scale", None)
    if u_scale is None:
        u_scale = 1.0

    disp_norm_val = args.disp_norm if args.disp_norm is not None else model_config.get('disp_norm', 64.0)
    udf_k = args.udf_k
    # CLI arg takes priority; fall back to model config, then default 1.0
    udf_channel_weight = args.udf_channel_weight if args.udf_channel_weight != 1.0 \
        else model_config.get('udf_channel_weight', 1.0)

    def transform_reals(t):
        """Map dataset-space reals to diffusion training space.

        1-channel mode [disp]:
          ch0: disp / disp_norm

        2-channel mode [udf, disp]:
          ch0: (1 - exp(-k * udf)) * udf_channel_weight
          ch1: disp / disp_norm

        3-channel layered mode [udf_obj1, udf_obj2, disp]:
          ch0: (1 - exp(-k * udf_obj1)) * udf_channel_weight
          ch1: (1 - exp(-k * udf_obj2)) * udf_channel_weight
          ch2: disp / disp_norm
        """
        if t.shape[1] == 3:
            # Layered per-object UDF mode
            if args.disable_udf_boundary_focus:
                udf1_t = t[:, 0:1] * udf_channel_weight
                udf2_t = t[:, 1:2] * udf_channel_weight
            else:
                udf1_t = (1.0 - torch.exp(-udf_k * t[:, 0:1])) * udf_channel_weight
                udf2_t = (1.0 - torch.exp(-udf_k * t[:, 1:2])) * udf_channel_weight
            disp_t = t[:, 2:3] / disp_norm_val
            return torch.cat([udf1_t, udf2_t, disp_t], dim=1)
        elif t.shape[1] == 1:
            # Disp-only mode
            return t / disp_norm_val
        else:
            if args.disable_udf_boundary_focus:
                udf_t = t[:, 0:1] * udf_channel_weight
            else:
                udf_t = (1.0 - torch.exp(-udf_k * t[:, 0:1])) * udf_channel_weight
            disp_t = t[:, 1:2] / disp_norm_val
            return torch.cat([udf_t, disp_t], dim=1)

    def inverse_transform_pred(pred):
        """Invert transform_reals.

        1-channel: returns (None, disp_px)
        2-channel: returns (udf_px, disp_px)
        3-channel: returns (udf1_px, udf2_px, disp_px)
        """
        if pred.shape[1] == 3:
            udf1_t = (pred[:, 0:1] / udf_channel_weight).clamp(0.0, 1.0 - 1e-6)
            udf2_t = (pred[:, 1:2] / udf_channel_weight).clamp(0.0, 1.0 - 1e-6)
            disp_t = pred[:, 2:3]
            udf1_px = -torch.log(1.0 - udf1_t) / udf_k * u_scale
            udf2_px = -torch.log(1.0 - udf2_t) / udf_k * u_scale
            disp_px = disp_t * disp_norm_val
            return udf1_px, udf2_px, disp_px
        elif pred.shape[1] == 1:
            # Disp-only mode
            return None, pred[:, 0:1] * disp_norm_val
        else:
            udf_t  = (pred[:, 0:1] / udf_channel_weight).clamp(0.0, 1.0 - 1e-6)
            disp_t = pred[:, 1:2]
            udf_px  = -torch.log(1.0 - udf_t) / udf_k * u_scale
            disp_px = disp_t * disp_norm_val
            return udf_px, disp_px

    def save_gray_png(arr, path, vmin=None, vmax=None):
        """
        Save a 2D float array as an 8-bit grayscale PNG.
        If vmin/vmax are not given, use per-image min/max normalization.
        """
        from PIL import Image
        import numpy as np

        arr = np.asarray(arr, dtype=np.float32)

        if vmin is None:
            vmin = float(np.nanmin(arr))
        if vmax is None:
            vmax = float(np.nanmax(arr))

        denom = max(vmax - vmin, 1e-8)
        arr_01 = np.clip((arr - vmin) / denom, 0.0, 1.0)
        arr_u8 = (arr_01 * 255.0).astype(np.uint8)

        Image.fromarray(arr_u8).save(path)


    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def evaluate():
        if not evaluate_enabled:
            return
        if accelerator.is_main_process:
            tqdm.write("Evaluating MSE on validation set...")

        sigmas = K.sampling.get_sigmas_karras(
            50, sigma_min, sigma_max, rho=7., device=device
        )

        # Accumulators
        mse_sum = 0.0
        udf_mse_sum = 0.0
        udf1_mse_sum = 0.0
        udf2_mse_sum = 0.0
        disp_mse_sum = 0.0
        pix_sum = 0.0
        pix_ch_sum = 0.0
        # EPE (mean abs error, real pixel units -- unlike the MSE above,
        # which is in normalized/transformed training space)
        ae_udf_sum = 0.0
        ae_udf2_sum = 0.0
        ae_disp_sum = 0.0

        save_root = Path(f"{args.name}_evalpreds") / f"step_{step:08d}"
        save_root.mkdir(parents=True, exist_ok=True) if accelerator.is_main_process else None

        global_idx = 0

        for batch in tqdm(val_dl, leave=False, disable=not accelerator.is_main_process):
            batch_elems = batch[image_key]
            reals, aug_cond = batch_elems[0], batch_elems[2]
            reals_t = transform_reals(reals)

            eval_extra_args = {"aug_cond": aug_cond}
            if model_wants_c2f:
                if not has_coarse_strip:
                    raise ValueError(
                        "Model has use_scc_c2f=True but the dataset wasn't built with "
                        "return_coarse_strip=True -- set it in val_config too.")
                idx = 4
                if has_amodal_masks:
                    idx += 1
                if has_disp_valid:
                    idx += 1
                eval_extra_args["scc_coarse_L"] = batch_elems[idx].to(reals_t.dtype)
                eval_extra_args["scc_coarse_R"] = batch_elems[idx + 1].to(reals_t.dtype)

            x = torch.randn_like(reals_t) * sigma_max
            if det_cfg:
                x_pred = model_ema.inner_model(
                    torch.zeros_like(reals_t),
                    reals_t.new_full([reals_t.shape[0]], det_cfg['sigma']), **eval_extra_args)
            else:
                x_pred = K.sampling.sample_dpmpp_2m_sde(
                    model_ema,
                    x,
                    sigmas,
                    extra_args=eval_extra_args,
                    eta=0.0,
                    solver_type="heun",
                    disable=True
                )

            # ---- MSE in transformed training space, separated by channel ----
            sq_err_eval = (x_pred - reals_t).pow(2)
            is_3ch_eval = (reals_t.shape[1] == 3)
            is_1ch_eval = (reals_t.shape[1] == 1)

            se_total = sq_err_eval.sum()
            if is_1ch_eval:
                # Disp-only: single channel is disparity
                se_udf  = torch.zeros(1, device=device)
                se_udf2 = None
                se_disp = sq_err_eval[:, 0:1].sum()
            elif is_3ch_eval:
                se_udf  = sq_err_eval[:, 0:1].sum()
                se_udf2 = sq_err_eval[:, 1:2].sum()
                se_disp = sq_err_eval[:, 2:3].sum()
            else:
                se_udf  = sq_err_eval[:, 0:1].sum()
                se_udf2 = None
                se_disp = sq_err_eval[:, 1:2].sum()

            npx_total = torch.tensor(
                [reals_t.numel()], device=device, dtype=torch.float32
            )
            npx_ch = torch.tensor(
                [reals_t[:, 0:1].numel()], device=device, dtype=torch.float32
            )

            se_total = accelerator.gather(se_total).sum().item()
            se_udf   = accelerator.gather(se_udf).sum().item()
            se_disp  = accelerator.gather(se_disp).sum().item()
            if se_udf2 is not None:
                se_udf2 = accelerator.gather(se_udf2).sum().item()
            npx_total = accelerator.gather(npx_total).sum().item()
            npx_ch    = accelerator.gather(npx_ch).sum().item()

            mse_sum += se_total
            udf_mse_sum += se_udf
            if se_udf2 is not None:
                udf2_mse_sum += se_udf2
            disp_mse_sum += se_disp
            pix_sum += npx_total
            pix_ch_sum += npx_ch

            # ---- EPE (mean abs error) in real pixel units ----
            # reals is dataset-space (pre-transform_reals), i.e. already in
            # real pixel/disparity units, so no inversion needed on the GT
            # side -- only the prediction needs inverse_transform_pred.
            inv = inverse_transform_pred(x_pred)
            if is_1ch_eval:
                ae_udf  = torch.zeros(1, device=device)
                ae_udf2 = None
                ae_disp = (inv[1] - reals[:, 0:1]).abs().sum()
            elif is_3ch_eval:
                udf1_px, udf2_px, disp_px = inv
                ae_udf  = (udf1_px - reals[:, 0:1]).abs().sum()
                ae_udf2 = (udf2_px - reals[:, 1:2]).abs().sum()
                ae_disp = (disp_px - reals[:, 2:3]).abs().sum()
            else:
                udf_px, disp_px = inv
                ae_udf  = (udf_px - reals[:, 0:1]).abs().sum()
                ae_udf2 = None
                ae_disp = (disp_px - reals[:, 1:2]).abs().sum()

            ae_udf  = accelerator.gather(ae_udf).sum().item()
            ae_disp = accelerator.gather(ae_disp).sum().item()
            if ae_udf2 is not None:
                ae_udf2 = accelerator.gather(ae_udf2).sum().item()

            ae_udf_sum  += ae_udf
            ae_disp_sum += ae_disp
            if ae_udf2 is not None:
                ae_udf2_sum += ae_udf2

            # ---- Save predictions as .npy and .png on rank 0 ----
            if accelerator.is_main_process:
                N = x_pred.size(0)

                for i in range(N):
                    if is_3ch_eval:
                        udf1_px, udf2_px, disp_px = inv
                        udf1_np = udf1_px[i, 0].detach().cpu().float().numpy().astype(np.float32)
                        udf2_np = udf2_px[i, 0].detach().cpu().float().numpy().astype(np.float32)
                        disp_np = disp_px[i, 0].detach().cpu().float().numpy().astype(np.float32)

                        np.save(save_root / f"val_{global_idx:06d}_pred_udf_obj1.npy", udf1_np)
                        np.save(save_root / f"val_{global_idx:06d}_pred_udf_obj2.npy", udf2_np)
                        np.save(save_root / f"val_{global_idx:06d}_pred_disp.npy", disp_np)
                        save_gray_png(udf1_np, save_root / f"val_{global_idx:06d}_pred_udf_obj1.png")
                        save_gray_png(udf2_np, save_root / f"val_{global_idx:06d}_pred_udf_obj2.png")
                        save_gray_png(disp_np, save_root / f"val_{global_idx:06d}_pred_disp.png",
                                      vmin=0.0, vmax=disp_norm_val)
                    else:
                        udf_px, disp_px = inv
                        disp_np = disp_px[i, 0].detach().cpu().float().numpy().astype(np.float32)
                        if udf_px is not None:
                            udf_np = udf_px[i, 0].detach().cpu().float().numpy().astype(np.float32)
                            np.save(save_root / f"val_{global_idx:06d}_pred_udf.npy", udf_np)
                            save_gray_png(udf_np, save_root / f"val_{global_idx:06d}_pred_udf.png")
                        np.save(save_root / f"val_{global_idx:06d}_pred_disp.npy", disp_np)
                        save_gray_png(disp_np, save_root / f"val_{global_idx:06d}_pred_disp.png",
                                      vmin=0.0, vmax=disp_norm_val)

                    global_idx += 1

        mse_mean = mse_sum / pix_sum
        disp_mse_mean = disp_mse_sum / pix_ch_sum
        disp_epe_mean = ae_disp_sum / pix_ch_sum
        is_disp_only = (udf_mse_sum == 0.0 and udf2_mse_sum == 0.0)
        if is_disp_only:
            udf_mse_mean = None
            udf2_mse_mean = None
            avg_udf_mse = None
            udf_epe_mean = None
            udf2_epe_mean = None
            avg_udf_epe = None
        else:
            udf_mse_mean = udf_mse_sum / pix_ch_sum
            udf_epe_mean = ae_udf_sum / pix_ch_sum
            if udf2_mse_sum > 0.0:
                udf2_mse_mean = udf2_mse_sum / pix_ch_sum
                udf2_epe_mean = ae_udf2_sum / pix_ch_sum
                avg_udf_mse = (udf_mse_mean + udf2_mse_mean) / 2
                avg_udf_epe = (udf_epe_mean + udf2_epe_mean) / 2
            else:
                udf2_mse_mean = None
                udf2_epe_mean = None
                avg_udf_mse = udf_mse_mean
                avg_udf_epe = udf_epe_mean

        if accelerator.is_main_process:
            if is_disp_only:
                tqdm.write(
                    f"Val MSE: {mse_mean:.10f} | "
                    f"Disp: {disp_mse_mean:.10f} (EPE {disp_epe_mean:.4f}px) "
                    f"(saved preds to: {save_root})"
                )
            elif udf2_mse_mean is not None:
                tqdm.write(
                    f"Val MSE: {mse_mean:.10f} | "
                    f"UDF1: {udf_mse_mean:.10f} (EPE {udf_epe_mean:.4f}px) | "
                    f"UDF2: {udf2_mse_mean:.10f} (EPE {udf2_epe_mean:.4f}px) | "
                    f"avg_UDF: {avg_udf_mse:.10f} (EPE {avg_udf_epe:.4f}px) | "
                    f"Disp: {disp_mse_mean:.10f} (EPE {disp_epe_mean:.4f}px) "
                    f"(saved preds to: {save_root})"
                )
            else:
                tqdm.write(
                    f"Val MSE: {mse_mean:.10f} | "
                    f"UDF: {udf_mse_mean:.10f} (EPE {udf_epe_mean:.4f}px) | "
                    f"Disp: {disp_mse_mean:.10f} (EPE {disp_epe_mean:.4f}px) "
                    f"(saved preds to: {save_root})"
                )

            if use_wandb:
                log_val = {
                    "val/loss": mse_mean,
                    "val/loss_disp": disp_mse_mean,
                    "val/epe_disp": disp_epe_mean,
                }
                if avg_udf_mse is not None:
                    log_val["val/loss_udf"] = avg_udf_mse
                    log_val["val/epe_udf"] = avg_udf_epe
                if udf2_mse_mean is not None:
                    log_val["val/loss_udf1"] = udf_mse_mean
                    log_val["val/loss_udf2"] = udf2_mse_mean
                    log_val["val/epe_udf1"] = udf_epe_mean
                    log_val["val/epe_udf2"] = udf2_epe_mean
                wandb.log(log_val, step=step)



    def save():
        accelerator.wait_for_everyone()
        filename = f'{args.name}_{step:08}.pth'
        if accelerator.is_main_process:
            tqdm.write(f'Saving to {filename}...')
        inner_model = unwrap(model.inner_model)
        inner_model_ema = unwrap(model_ema.inner_model)
        obj = {
            'config': config,
            'model': inner_model.state_dict(),
            'model_ema': inner_model_ema.state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'ema_sched': ema_sched.state_dict(),
            'epoch': epoch,
            'step': step,
            'gns_stats': gns_stats.state_dict() if gns_stats is not None else None,
            'ema_stats': ema_stats,
            'demo_gen': demo_gen.get_state(),
            'elapsed': elapsed,
        }
        accelerator.save(obj, filename)
        if accelerator.is_main_process:
            state_obj = {'latest_checkpoint': filename}
            json.dump(state_obj, open(state_path, 'w'))
        if args.wandb_save_model and use_wandb:
            wandb.save(filename)

    # Determined once from the dataset object (not from per-batch tuple length,
    # which becomes ambiguous once both amodal_masks and disp_valid can be
    # appended) -- see FoJStereoDataset.__getitem__'s tuple-ordering comment.
    # Computed here (before the --evaluate-only branch, not after) since
    # evaluate() needs these too and --evaluate-only can call it before
    # reaching the training loop below.
    has_amodal_masks  = getattr(train_set, 'use_amodal', False)
    has_disp_valid    = getattr(train_set, 'return_disp_valid', False)
    has_coarse_strip  = getattr(train_set, 'return_coarse_strip', False)
    model_wants_c2f   = getattr(unwrap(model.inner_model), 'use_scc_c2f', False)

    if args.evaluate_only:
        if not evaluate_enabled:
            raise ValueError('--evaluate-only requested but evaluation is disabled')
        evaluate()
        return

    losses_since_last_print        = []
    udf_losses_since_last_print    = []
    disp_losses_since_last_print   = []
    amodal_losses_since_last_print = []
    udf1_losses_since_last_print   = []
    udf2_losses_since_last_print   = []
    distill_losses_since_last_print = []

    def _masked_ch_mean(x_ch, valid):
        """x_ch, valid: (N,1,H,W). Mean over spatial dims, excluding invalid
        pixels when `valid` is given (valid=None -> plain mean, old behavior)."""
        if valid is None:
            return x_ch.flatten(1).mean(1)
        v = valid.flatten(1)
        return (x_ch.flatten(1) * v).sum(1) / v.sum(1).clamp_min(1.0)

    try:
        while True:
            for batch in tqdm(train_dl, smoothing=0.1, disable=not accelerator.is_main_process):
                if device.type == 'cuda':
                    start_timer = torch.cuda.Event(enable_timing=True)
                    end_timer = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_timer.record()
                else:
                    start_timer = time.time()

                with accelerator.accumulate(model):
                    batch_elems = batch[image_key]
                    reals, aug_cond = batch_elems[0], batch_elems[2]
                    _next_idx = 4
                    amodal_gt = None
                    if args.amodal_loss_weight > 0.0 and has_amodal_masks:
                        amodal_gt = batch_elems[_next_idx].float()
                        _next_idx += 1
                    disp_valid = None
                    if has_disp_valid:
                        disp_valid = batch_elems[_next_idx].to(reals.dtype)
                        _next_idx += 1
                    scc_coarse_L = scc_coarse_R = None
                    if has_coarse_strip:
                        scc_coarse_L = batch_elems[_next_idx]
                        _next_idx += 1
                        scc_coarse_R = batch_elems[_next_idx]
                        _next_idx += 1

                    reals_t = transform_reals(reals)
                    class_cond, extra_args = None, {}
                    if num_classes:
                        class_cond = batch[class_key]
                        drop = torch.rand(class_cond.shape, device=class_cond.device)
                        class_cond.masked_fill_(drop < cond_dropout_rate, num_classes)
                        extra_args['class_cond'] = class_cond
                    noise = torch.randn_like(reals_t)
                    with K.utils.enable_stratified_accelerate(accelerator, disable=args.gns):
                        sigma = sample_density([reals_t.shape[0]], device=device)

                    # ── EDM loss (manual, supports per-channel logging + fg weighting) ──
                    sigma_4d = K.utils.append_dims(sigma, reals_t.ndim)
                    c_skip, c_out, c_in = [K.utils.append_dims(x, reals_t.ndim)
                                           for x in model.get_scalings(sigma)]
                    c_weight = model.weighting(sigma)
                    noised = reals_t + noise * sigma_4d
                    if det_cfg:
                        # deterministic regression: zero input, fixed sigma, predict x0 directly
                        sigma = torch.full_like(sigma, det_cfg['sigma'])
                        c_skip, c_out, c_in = (torch.zeros_like(c_skip), torch.ones_like(c_out),
                                               torch.ones_like(c_in))
                        c_weight = torch.ones_like(c_weight)
                        noised = torch.zeros_like(reals_t)
                    want_scc = (args.use_scc or args.use_scc_native) and args.scc_disp_weight > 0.0
                    scc_disp = scc_valid = scc_dxy = None

                    # ── RAFT-disp conditioning: use GT disp as training proxy ──────────────
                    if args.use_raft_disp:
                        _nch = reals.shape[1]
                        _disp_ch = 2 if _nch == 3 else (0 if _nch == 1 else 1)
                        raft_disp_cond = reals[:, _disp_ch:_disp_ch+1].clone().float()
                        if args.raft_disp_noise_std > 0:
                            raft_disp_cond = raft_disp_cond + torch.randn_like(raft_disp_cond) * args.raft_disp_noise_std
                        # Pass raw pixels — model.forward normalizes by raft_disp_norm internally.
                        if args.raft_disp_dropout > 0:
                            keep = (torch.rand(raft_disp_cond.shape[0], 1, 1, 1,
                                               device=raft_disp_cond.device) > args.raft_disp_dropout).float()
                            raft_disp_cond = raft_disp_cond * keep
                        extra_args['raft_disp'] = raft_disp_cond.to(reals_t.dtype)

                    if model_wants_c2f:
                        if scc_coarse_L is None:
                            raise ValueError(
                                "Model has use_scc_c2f=True but the dataset wasn't built with "
                                "return_coarse_strip=True -- set it in the dataset config.")
                        extra_args['scc_coarse_L'] = scc_coarse_L.to(reals_t.dtype)
                        extra_args['scc_coarse_R'] = scc_coarse_R.to(reals_t.dtype)

                    with K.models.checkpointing(args.checkpointing):
                        if amodal_gt is not None:
                            pred, amodal_pred = model.inner_model(noised * c_in, sigma,
                                                                  aug_cond=aug_cond,
                                                                  return_amodal=True,
                                                                  **extra_args)
                        elif want_scc:
                            pred, scc_aux = model.inner_model(noised * c_in, sigma,
                                                              aug_cond=aug_cond,
                                                              return_scc=True,
                                                              **extra_args)
                            scc_disp, scc_valid = scc_aux["disp"], scc_aux["valid"]
                            scc_dxy = scc_aux.get("dxy")
                        else:
                            pred = model.inner_model(noised * c_in, sigma,
                                                     aug_cond=aug_cond, **extra_args)
                    target = (reals_t - c_skip * noised) / c_out
                    sq_err = (pred - target) ** 2                          # (N,2,H,W)

                    is_3ch = (reals_t.shape[1] == 3)
                    is_1ch = (reals_t.shape[1] == 1)
                    if args.fg_alpha > 0.0:
                        # foreground = where disparity > 0 (ch2 in 3-ch, ch1 in 2-ch, ch0 in 1-ch)
                        disp_ch = 2 if is_3ch else (0 if is_1ch else 1)
                        fg_mask = (reals[:, disp_ch:disp_ch+1] > 0.0005).float()
                        spatial_w = (1.0 + args.fg_alpha * fg_mask).expand_as(reals_t)
                        sq_err_w = sq_err * spatial_w
                    else:
                        sq_err_w = sq_err

                    # ── Boundary-weighted disparity loss ──────────────────────────────
                    # Upweights disp MSE near UDF=0 (object boundaries) so the model
                    # is penalised more heavily for disparity errors at depth edges.
                    if args.bw_enable and not is_1ch:
                        udf_px = reals[:, 0:1] * u_scale          # ch0 UDF in pixels
                        if is_3ch:
                            # use nearest boundary of either object
                            udf_px = torch.minimum(udf_px, reals[:, 1:2] * u_scale)
                        if args.bw_soft_beta_px > 0:
                            bw_map = 1.0 + args.bw_alpha * torch.exp(
                                -(udf_px ** 2) / (2.0 * args.bw_soft_beta_px ** 2))
                        else:
                            bw_map = 1.0 + args.bw_alpha * (udf_px < args.bw_tau_px).float()
                        disp_w = (1.0 - args.bw_lambda) + args.bw_lambda * bw_map
                        # build a per-channel weight: 1 for UDF channels, disp_w for disp
                        disp_ch_bw = 2 if is_3ch else 1
                        ch_w = torch.ones_like(sq_err_w)
                        ch_w[:, disp_ch_bw:disp_ch_bw+1] = disp_w
                        sq_err_w = sq_err_w * ch_w

                    # per-channel losses (N,) — mean over spatial, weighted by sigma
                    if is_1ch:
                        # Disp-only mode: single channel is disparity
                        loss_udf1_per = None
                        loss_udf2_per = None
                        loss_udf_per  = None
                        loss_disp_per = _masked_ch_mean(sq_err_w[:, 0:1], disp_valid) * c_weight
                        losses = loss_disp_per
                    elif is_3ch:
                        loss_udf1_per = sq_err_w[:, 0:1].flatten(1).mean(1) * c_weight
                        loss_udf2_per = sq_err_w[:, 1:2].flatten(1).mean(1) * c_weight
                        loss_disp_per = _masked_ch_mean(sq_err_w[:, 2:3], disp_valid) * c_weight
                        loss_udf_per  = (loss_udf1_per + loss_udf2_per) / 2
                        if args.loss_mode == 'joint':
                            losses = (loss_udf1_per + loss_udf2_per + loss_disp_per) / 3
                        elif args.loss_mode == 'udf_only':
                            losses = (loss_udf1_per + loss_udf2_per) / 2
                        elif args.loss_mode == 'disp_only':
                            losses = loss_disp_per
                        else:
                            raise ValueError(f"Unknown loss_mode: {args.loss_mode}")
                    else:
                        loss_udf1_per = None
                        loss_udf2_per = None
                        loss_udf_per  = sq_err_w[:, 0:1].flatten(1).mean(1) * c_weight
                        loss_disp_per = _masked_ch_mean(sq_err_w[:, 1:2], disp_valid) * c_weight
                        if args.loss_mode == 'joint':
                            losses = (loss_udf_per + loss_disp_per) / 2
                        elif args.loss_mode == 'udf_only':
                            losses = loss_udf_per
                        elif args.loss_mode == 'disp_only':
                            losses = loss_disp_per
                        else:
                            raise ValueError(f"Unknown loss_mode: {args.loss_mode}")

                    # ── Distillation: blend teacher-matching loss into the ground-truth loss ──
                    # Runs the frozen teacher on the SAME noised input/sigma/conditioning the
                    # student just saw, and pulls the student's raw (pre-c_skip/c_out) output
                    # toward the teacher's, blended with the ground-truth loss above via
                    # --distill-weight. No effect at all unless --teacher-checkpoint was set.
                    # extra_args is reused as-is for the teacher call -- valid because the
                    # teacher is assumed to be the same architecture family (same conditioning
                    # inputs expected) as the student, just a different size.
                    if teacher_model is not None:
                        with torch.no_grad():
                            pred_teacher = teacher_model.inner_model(noised * c_in, sigma,
                                                                     aug_cond=aug_cond, **extra_args)
                        sq_err_distill = (pred - pred_teacher.detach()) ** 2
                        loss_distill_per = sq_err_distill.flatten(1).mean(1) * c_weight
                        losses = (1.0 - args.distill_weight) * losses + args.distill_weight * loss_distill_per
                        loss_distill = accelerator.gather(loss_distill_per).mean().item()
                    else:
                        loss_distill = 0.0

                    # ── SCC disparity supervision (soft-argmax disparity vs GT disp) ──
                    if want_scc and scc_disp is not None:
                        gt_disp_ch = 2 if is_3ch else (0 if is_1ch else 1)
                        # SCC predicts ABSOLUTE disparity, so supervise on the absolute GT even
                        # when the diffusion channel has been reparameterized to a residual.
                        disp_gt_full = reals[:, gt_disp_ch:gt_disp_ch+1]             # pixels (B,1,H,W)
                        if disp_valid is not None:
                            # Masked pooling: a token straddling valid/invalid pixels must
                            # average only the valid ones, or the invalid-fill (0) drags
                            # gt_small toward a false value even where w partially downweights it.
                            dv_small = F.adaptive_avg_pool2d(disp_valid, scc_disp.shape[-2:])
                            gt_small = (F.adaptive_avg_pool2d(disp_gt_full * disp_valid, scc_disp.shape[-2:])
                                        / dv_small.clamp_min(1e-6)).to(scc_disp.dtype)
                        else:
                            dv_small = None
                            gt_small = F.adaptive_avg_pool2d(disp_gt_full, scc_disp.shape[-2:]).to(scc_disp.dtype)
                        w = (scc_valid if scc_valid is not None else torch.ones_like(scc_disp)).to(scc_disp.dtype)
                        if dv_small is not None:
                            w = w * dv_small.to(scc_disp.dtype)
                        scc_l1 = F.smooth_l1_loss(scc_disp, gt_small, reduction='none') * w
                        scc_loss_per = scc_l1.flatten(1).sum(1) / w.flatten(1).sum(1).clamp_min(1.0)

                        # ── scc-plane: supervise (dx, dy) against GT disparity gradients ──
                        # gt_small is px disparity on the token grid; adjacent tokens are
                        # px_per_tok pixels apart, so GT gradient (px/px) = finite diff / px_per_tok.
                        if scc_dxy is not None:
                            px_per_tok = disp_gt_full.shape[-1] / scc_disp.shape[-1]
                            gt_gx = (gt_small[..., :, 1:] - gt_small[..., :, :-1]) / px_per_tok
                            gt_gy = (gt_small[..., 1:, :] - gt_small[..., :-1, :]) / px_per_tok
                            wx = torch.minimum(w[..., :, 1:], w[..., :, :-1])
                            wy = torch.minimum(w[..., 1:, :], w[..., :-1, :])
                            gx_l1 = F.smooth_l1_loss(scc_dxy[:, 0:1, :, 1:], gt_gx, reduction='none') * wx
                            gy_l1 = F.smooth_l1_loss(scc_dxy[:, 1:2, 1:, :], gt_gy, reduction='none') * wy
                            grad_loss_per = (gx_l1.flatten(1).sum(1) / wx.flatten(1).sum(1).clamp_min(1.0)
                                             + gy_l1.flatten(1).sum(1) / wy.flatten(1).sum(1).clamp_min(1.0))
                            scc_loss_per = scc_loss_per + grad_loss_per

                        losses = losses + args.scc_disp_weight * scc_loss_per
                        loss_scc = accelerator.gather(scc_loss_per).mean().item()
                    else:
                        loss_scc = 0.0

                    # ── Amodal auxiliary loss (BCE on per-object mask prediction) ──
                    if amodal_gt is not None:
                        if args.amodal_intersection_boost > 0.0:
                            # Upweight pixels where both objects overlap
                            intersection = (amodal_gt[:, 0:1] > 0.5) & (amodal_gt[:, 1:2] > 0.5)
                            pixel_w = (1.0 + args.amodal_intersection_boost
                                       * intersection.float()).expand_as(amodal_pred)
                            bce_raw = F.binary_cross_entropy(
                                amodal_pred, amodal_gt, weight=pixel_w, reduction='none'
                            )
                        else:
                            bce_raw = F.binary_cross_entropy(
                                amodal_pred, amodal_gt, reduction='none'
                            )
                        loss_amodal_per = bce_raw.flatten(1).mean(1) * args.amodal_loss_weight
                        losses = losses + loss_amodal_per
                        loss_amodal = accelerator.gather(loss_amodal_per).mean().item()
                    else:
                        loss_amodal = 0.0

                    loss      = accelerator.gather(losses).mean().item()
                    loss_udf  = accelerator.gather(loss_udf_per).mean().item() if loss_udf_per is not None else 0.0
                    loss_disp = accelerator.gather(loss_disp_per).mean().item()
                    if loss_udf1_per is not None:
                        loss_udf1 = accelerator.gather(loss_udf1_per).mean().item()
                        loss_udf2 = accelerator.gather(loss_udf2_per).mean().item()
                    else:
                        loss_udf1 = loss_udf2 = None
                    losses_since_last_print.append(loss)
                    udf_losses_since_last_print.append(loss_udf)
                    disp_losses_since_last_print.append(loss_disp)
                    amodal_losses_since_last_print.append(loss_amodal)
                    if loss_udf1 is not None:
                        udf1_losses_since_last_print.append(loss_udf1)
                        udf2_losses_since_last_print.append(loss_udf2)
                    if teacher_model is not None:
                        distill_losses_since_last_print.append(loss_distill)
                    accelerator.backward(losses.mean())

                    if args.gns:
                        sq_norm_small_batch, sq_norm_large_batch = gns_stats_hook.get_stats()
                        gns_stats.update(sq_norm_small_batch, sq_norm_large_batch, reals.shape[0], reals.shape[0] * accelerator.num_processes)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                    sched.step()
                    opt.zero_grad()

                    # ema_decay = ema_sched.get_value()
                    # K.utils.ema_update_dict(ema_stats, {'loss': loss}, ema_decay ** (1 / args.grad_accum_steps))
                    # if accelerator.sync_gradients:
                    #     K.utils.ema_update(model, model_ema, ema_decay)
                    #     ema_sched.step()

                    ema_decay = ema_sched.get_value()
                    K.utils.ema_update_dict(ema_stats, {'loss': loss}, ema_decay ** (1 / args.grad_accum_steps))
                    if accelerator.sync_gradients:
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()

                if device.type == 'cuda':
                    end_timer.record()
                    torch.cuda.synchronize()
                    elapsed += start_timer.elapsed_time(end_timer) / 1000
                else:
                    elapsed += time.time() - start_timer

                if step % 25 == 0:
                    avg_total  = sum(losses_since_last_print) / len(losses_since_last_print)
                    avg_udf    = sum(udf_losses_since_last_print) / len(udf_losses_since_last_print)
                    avg_disp   = sum(disp_losses_since_last_print) / len(disp_losses_since_last_print)
                    avg_amodal = sum(amodal_losses_since_last_print) / len(amodal_losses_since_last_print)
                    avg_udf1   = sum(udf1_losses_since_last_print) / len(udf1_losses_since_last_print) if udf1_losses_since_last_print else None
                    avg_udf2   = sum(udf2_losses_since_last_print) / len(udf2_losses_since_last_print) if udf2_losses_since_last_print else None
                    avg_distill = (sum(distill_losses_since_last_print) / len(distill_losses_since_last_print)
                                   if distill_losses_since_last_print else None)
                    losses_since_last_print.clear()
                    udf_losses_since_last_print.clear()
                    disp_losses_since_last_print.clear()
                    amodal_losses_since_last_print.clear()
                    udf1_losses_since_last_print.clear()
                    udf2_losses_since_last_print.clear()
                    distill_losses_since_last_print.clear()
                    ema_loss = ema_stats['loss']
                    if accelerator.is_main_process:
                        if avg_udf1 is not None:
                            msg = (f'Epoch: {epoch}, step: {step} | '
                                   f'loss: {avg_total:g}  udf1: {avg_udf1:g}  udf2: {avg_udf2:g}  disp: {avg_disp:g}')
                        else:
                            msg = (f'Epoch: {epoch}, step: {step} | '
                                   f'loss: {avg_total:g}  udf: {avg_udf:g}  disp: {avg_disp:g}')
                        if args.amodal_loss_weight > 0.0:
                            msg += f'  amodal: {avg_amodal:g}'
                        if teacher_model is not None and avg_distill is not None:
                            msg += f'  distill: {avg_distill:g}'
                        if args.use_scc or args.use_scc_native:
                            msg += f'  scc: {loss_scc:g}'
                            if args.use_scc:
                                msg += f'  gate: {float(unwrap(model.inner_model).scc_gate):g}'
                        msg += f' | ema_loss: {ema_loss:g}'
                        if args.gns:
                            msg += f'  gns: {gns_stats.get_gns():g}'
                        tqdm.write(msg)

                # if use_wandb:
                #     log_dict = {
                #         'epoch': epoch,
                #         'loss': loss,
                #         'lr': sched.get_last_lr()[0],
                #         'ema_decay': ema_decay,
                #     }
                #     if args.gns:
                #         log_dict['gradient_noise_scale'] = gns_stats.get_gns()
                #     wandb.log(log_dict, step=step)

                if use_wandb:
                    log_dict = {
                        'train/loss':      loss,
                        'train/loss_udf':  loss_udf,
                        'train/loss_disp': loss_disp,
                        'train/lr':        sched.get_last_lr()[0],
                        'train/ema_decay': ema_decay,
                        'epoch':           epoch,
                    }
                    if loss_udf1 is not None:
                        log_dict['train/loss_udf1'] = loss_udf1
                        log_dict['train/loss_udf2'] = loss_udf2
                    if args.amodal_loss_weight > 0.0:
                        log_dict['train/loss_amodal'] = loss_amodal
                    if teacher_model is not None:
                        log_dict['train/loss_distill'] = loss_distill
                        log_dict['train/distill_weight'] = args.distill_weight
                    if args.use_scc or args.use_scc_native:
                        log_dict['train/loss_scc'] = loss_scc
                        if args.use_scc:
                            log_dict['train/scc_gate'] = float(unwrap(model.inner_model).scc_gate)
                    if args.gns:
                        log_dict['train/gns'] = gns_stats.get_gns()
                    wandb.log(log_dict, step=step)
                step += 1

                if step % args.demo_every == 0:
                    demo()

                if evaluate_enabled and step > 0 and step % args.evaluate_every == 0:
                    evaluate()

                if step == args.end_step or (step > 0 and step % args.save_every == 0):
                    save()

                if step == args.end_step:
                    if accelerator.is_main_process:
                        tqdm.write('Done!')
                    return

            epoch += 1
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
