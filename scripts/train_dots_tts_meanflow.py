#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from einops import rearrange
from torch.optim import AdamW
from train_dots_tts import DotsTtsTrainingRun
from transformers import get_cosine_schedule_with_warmup

from dots_tts.config import app as app_config
from dots_tts.data import builders as data_module
from dots_tts.models.dots_tts import model as dots_tts_model
from dots_tts.models.dots_tts.config import MeanFlowConfig
from dots_tts.models.dots_tts.core import DotsTtsForwardOutput
from dots_tts.modules.backbone.dit import DiT
from dots_tts.training import checkpoint as train_checkpoint
from dots_tts.training import utils as train_utils
from dots_tts.utils import util as util_module

_ALLOWED_TEACHER_SOLVERS = ("euler", "midpoint", "rk4")
_ALLOWED_CFG_DISTILL_MODES = ("natural", "fused")
_ALLOWED_ANCHOR_TARGETS = ("formula", "teacher")


@dataclass(frozen=True, slots=True)
class MeanFlowSettings:
    teacher_model_path: str | None
    teacher_steps: int = 8
    teacher_solver: str = "euler"
    cfg_distill_mode: str = "fused"
    distill_cfg_scale: float = 1.2
    anchor_prob: float = 0.5
    anchor_target: str = "formula"
    time_sampling_mean: float = -0.4
    time_sampling_std: float = 1.0
    train_all_parameters: bool = False

    def __post_init__(self) -> None:
        if int(self.teacher_steps) <= 0:
            raise ValueError("teacher_steps must be positive.")
        if self.teacher_solver not in _ALLOWED_TEACHER_SOLVERS:
            raise ValueError(
                f"teacher_solver must be one of {_ALLOWED_TEACHER_SOLVERS}, "
                f"got {self.teacher_solver!r}."
            )
        if self.cfg_distill_mode not in _ALLOWED_CFG_DISTILL_MODES:
            raise ValueError(
                "cfg_distill_mode must be one of "
                f"{_ALLOWED_CFG_DISTILL_MODES}, got {self.cfg_distill_mode!r}."
            )
        if self.anchor_target not in _ALLOWED_ANCHOR_TARGETS:
            raise ValueError(
                f"anchor_target must be one of {_ALLOWED_ANCHOR_TARGETS}, "
                f"got {self.anchor_target!r}."
            )
        if not 0.0 <= float(self.anchor_prob) <= 1.0:
            raise ValueError("anchor_prob must be in [0, 1].")

    def to_dict(self) -> dict[str, Any]:
        return {
            "teacher_model_path": self.teacher_model_path,
            "teacher_steps": int(self.teacher_steps),
            "teacher_solver": self.teacher_solver,
            "cfg_distill_mode": self.cfg_distill_mode,
            "distill_cfg_scale": float(self.distill_cfg_scale),
            "anchor_prob": float(self.anchor_prob),
            "anchor_target": self.anchor_target,
            "time_sampling_mean": float(self.time_sampling_mean),
            "time_sampling_std": float(self.time_sampling_std),
            "train_all_parameters": bool(self.train_all_parameters),
        }


def enable_meanflow_student(model: dots_tts_model.DotsTtsModel) -> None:
    meanflow_config = MeanFlowConfig(enabled=True, use_duration_embedding=True)
    model.config.meanflow = meanflow_config
    model.core.meanflow_config = meanflow_config
    model.core.mode = "meanflow"

    old_dit = model.core.velocity_field_predictor
    if getattr(old_dit, "duration_embedder", None) is not None:
        return

    new_dit = DiT(
        in_dim=model.core.fm_hidden_size,
        out_dim=model.core.latent_dim,
        transformer_config=model.core.config.DiT,
        mode="meanflow",
    )
    missing_keys, unexpected_keys = new_dit.load_state_dict(
        old_dit.state_dict(),
        strict=False,
    )
    missing_keys = [
        key for key in missing_keys if not key.startswith("duration_embedder.")
    ]
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Failed to initialize MeanFlow DiT from the pretrained flow-matching "
            f"DiT: missing={missing_keys[:5]} unexpected={unexpected_keys[:5]}"
        )
    duration_output = new_dit.duration_embedder.mlp[-1]
    nn.init.zeros_(duration_output.weight)
    nn.init.zeros_(duration_output.bias)
    model.core.velocity_field_predictor = new_dit


class MeanFlowDotsTtsModel(nn.Module):
    def __init__(
        self,
        student: dots_tts_model.DotsTtsModel,
        settings: MeanFlowSettings,
    ):
        super().__init__()
        self.student = student
        self.settings = settings
        self._teacher_holder: dict[str, dots_tts_model.DotsTtsModel] = {}

    @property
    def config(self):
        return self.student.config

    @property
    def tokenizer(self):
        return self.student.tokenizer

    @property
    def teacher(self) -> dots_tts_model.DotsTtsModel:
        teacher = self._teacher_holder.get("model")
        if teacher is None:
            raise RuntimeError("MeanFlow teacher model has not been initialized.")
        return teacher

    def set_teacher(self, teacher: dots_tts_model.DotsTtsModel) -> None:
        for param in teacher.parameters():
            param.requires_grad_(False)
        teacher.eval()
        self._teacher_holder["model"] = teacher

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        teacher = self._teacher_holder.get("model")
        if teacher is not None:
            self._teacher_holder["model"] = teacher.to(*args, **kwargs)
            self._teacher_holder["model"].eval()
        return self

    def cuda(self, device=None):
        super().cuda(device)
        teacher = self._teacher_holder.get("model")
        if teacher is not None:
            self._teacher_holder["model"] = teacher.cuda(device).eval()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        teacher = self._teacher_holder.get("model")
        if teacher is not None:
            teacher.eval()
        return self

    def prepare_training_batch(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.student.prepare_training_batch(data)

    def save_pretrained(self, save_directory: str | Path) -> Path:
        return self.student.save_pretrained(save_directory)

    def load_pretrained_weights(
        self, pretrained_model_name_or_path: str | Path
    ) -> None:
        self.student.load_pretrained_weights(pretrained_model_name_or_path)

    def set_cfg_droprate(
        self,
        cfg_droprate: float | None = None,
        xvec_drop_rate: float | None = None,
    ) -> None:
        self.student.set_cfg_droprate(
            cfg_droprate=cfg_droprate,
            xvec_drop_rate=xvec_drop_rate,
        )

    @torch.no_grad()
    def compute_teacher_meanflow_target(
        self,
        *,
        xt: torch.Tensor,
        t: torch.Tensor,
        delta_t: torch.Tensor,
        prefix_data: dict[str, Any],
        g_cond: torch.Tensor | None,
        cfg_distill: bool,
        uncond_prefix_data: dict[str, Any] | None,
        uncond_g_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        teacher_core = self.teacher.core
        teacher_dit = teacher_core.velocity_field_predictor
        io_helper = teacher_core.io_helper
        noisy_proj = teacher_core.coordinate_proj
        n_steps = int(self.settings.teacher_steps)
        solver = self.settings.teacher_solver
        cfg_scale = float(self.settings.distill_cfg_scale)

        if solver not in _ALLOWED_TEACHER_SOLVERS:
            raise ValueError(f"Unsupported teacher solver: {solver!r}.")

        device = xt.device
        batch_size = xt.size(0)
        latent_lens = prefix_data["latent_lens"]
        latent_patch_size = int(prefix_data["latent_patch_size"])
        anchor_mask = delta_t.float() == 0

        autocast_device = device.type if device.type in {"cuda", "xpu"} else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=False):
            z = xt.float()
            cur_t = t.float()
            safe_dt = delta_t.float().clamp(min=1e-6)
            step_dt = safe_dt / n_steps

            def evaluate(z_in: torch.Tensor, t_val: torch.Tensor) -> torch.Tensor:
                fm_seq = io_helper.replace_noise_latents_in_fm_seq(
                    prefix_data,
                    z_in.to(xt.dtype),
                    noisy_proj,
                ).float()
                vt = teacher_dit(
                    x=fm_seq,
                    timesteps=t_val,
                    pos_ids=prefix_data["fm_pos_ids"],
                    mask=prefix_data["fm_seq_mask"],
                    attn_mask=prefix_data["fm_attn_mask"],
                    g_cond=None if g_cond is None else g_cond.float(),
                )
                pred = io_helper.get_dit_outputs(
                    pred_v=vt,
                    fm_prefix_lengths=prefix_data["fm_prefix_lengths"],
                    fm_gen_lengths=prefix_data["fm_gen_lengths"],
                    fm_gen_patch_size=prefix_data["fm_gen_patch_size"],
                    latent_patch_size=prefix_data["latent_patch_size"],
                )

                if cfg_distill:
                    if uncond_prefix_data is None:
                        raise RuntimeError(
                            "CFG distillation requires an uncond prefix."
                        )
                    fm_seq_u = io_helper.replace_noise_latents_in_fm_seq(
                        uncond_prefix_data,
                        z_in.to(xt.dtype),
                        noisy_proj,
                    ).float()
                    vt_u = teacher_dit(
                        x=fm_seq_u,
                        timesteps=t_val,
                        pos_ids=uncond_prefix_data["fm_pos_ids"],
                        mask=uncond_prefix_data["fm_seq_mask"],
                        attn_mask=uncond_prefix_data["fm_attn_mask"],
                        g_cond=None if uncond_g_cond is None else uncond_g_cond.float(),
                    )
                    pred_u = io_helper.get_dit_outputs(
                        pred_v=vt_u,
                        fm_prefix_lengths=uncond_prefix_data["fm_prefix_lengths"],
                        fm_gen_lengths=uncond_prefix_data["fm_gen_lengths"],
                        fm_gen_patch_size=uncond_prefix_data["fm_gen_patch_size"],
                        latent_patch_size=uncond_prefix_data["latent_patch_size"],
                    )
                    pred = pred + cfg_scale * (pred - pred_u)
                return rearrange(pred, "n p d -> (n p) d")

            v_init_flat = evaluate(z, cur_t)

            def apply_velocity(
                z_cur: torch.Tensor,
                v_flat: torch.Tensor,
                *,
                dt_factor: float,
            ) -> torch.Tensor:
                new_z = z_cur.clone()
                offset = 0
                for batch_idx in range(batch_size):
                    length = int(latent_lens[batch_idx].item())
                    if length <= 0:
                        continue
                    if not bool(anchor_mask[batch_idx].item()):
                        new_z[batch_idx, :length, :] = z_cur[
                            batch_idx, :length, :
                        ] + v_flat[offset : offset + length, :] * (
                            step_dt[batch_idx] * float(dt_factor)
                        )
                    offset += length
                return new_z

            if solver == "euler":
                v_flat = v_init_flat
                for step in range(n_steps):
                    if step > 0:
                        v_flat = evaluate(z, cur_t)
                    z = apply_velocity(z, v_flat, dt_factor=1.0)
                    cur_t = cur_t + step_dt
            elif solver == "midpoint":
                for step in range(n_steps):
                    k1 = v_init_flat if step == 0 else evaluate(z, cur_t)
                    z_mid = apply_velocity(z, k1, dt_factor=0.5)
                    k2 = evaluate(z_mid, cur_t + 0.5 * step_dt)
                    z = apply_velocity(z, k2, dt_factor=1.0)
                    cur_t = cur_t + step_dt
            else:
                for step in range(n_steps):
                    k1 = v_init_flat if step == 0 else evaluate(z, cur_t)
                    z1 = apply_velocity(z, k1, dt_factor=0.5)
                    k2 = evaluate(z1, cur_t + 0.5 * step_dt)
                    z2 = apply_velocity(z, k2, dt_factor=0.5)
                    k3 = evaluate(z2, cur_t + 0.5 * step_dt)
                    z3 = apply_velocity(z, k3, dt_factor=1.0)
                    k4 = evaluate(z3, cur_t + step_dt)
                    z = apply_velocity(
                        z,
                        (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0,
                        dt_factor=1.0,
                    )
                    cur_t = cur_t + step_dt

            mean_velocity = (z - xt.float()) / safe_dt.view(-1, 1, 1)
            target_chunks = []
            offset = 0
            for batch_idx in range(batch_size):
                length = int(latent_lens[batch_idx].item())
                if length <= 0:
                    continue
                if bool(anchor_mask[batch_idx].item()):
                    target_b = v_init_flat[offset : offset + length, :]
                else:
                    target_b = mean_velocity[batch_idx, :length, :]
                target_chunks.append(
                    rearrange(target_b, "(n p) d -> n p d", p=latent_patch_size)
                )
                offset += length
            if not target_chunks:
                raise RuntimeError("Teacher rollout produced no MeanFlow target.")
            return torch.cat(target_chunks, dim=0).to(xt.dtype)

    def forward(self, data: dict[str, Any]):
        loss_masks = data["loss_masks"]
        processed = self.student.prepare_training_inputs(data)
        processed["input_span_mask"] = data["input_span_mask"]
        processed["output_span_mask"] = data["output_span_mask"]
        outputs = self.meanflow_forward(processed)
        return self.student._compute_loss_terms(
            outputs,
            labels=processed["labels"],
            loss_masks=loss_masks,
        )

    def meanflow_forward(self, data: dict[str, Any]) -> DotsTtsForwardOutput:
        core = self.student.core
        input_ids: torch.Tensor = data["input_ids"]
        input_ids_lengths: torch.Tensor = data["input_ids_lengths"]
        input_span_mask: torch.Tensor = data["input_span_mask"]
        output_span_mask: torch.Tensor = data["output_span_mask"]
        batch_size = input_ids.size(0)
        device = input_ids.device

        latents: torch.Tensor | None = data.get("latents")
        latents_sampled: torch.Tensor | None = data.get("latents_sampled")
        latent_lengths: torch.Tensor | None = data.get("latent_lengths")
        has_latents = latents is not None or latents_sampled is not None

        if has_latents:
            if latents_sampled is None:
                latents_sampled = core.io_helper.sample_from_latent(latents)
            patch_embeddings = core.patch_encoder(
                latents_sampled, x_lens=latent_lengths
            )
            valid_patch_counts = latent_lengths // core.latent_patch_size
            latents_sampled = core.io_helper.normalize(latents_sampled)
        else:
            latents_sampled = None
            patch_embeddings = None
            valid_patch_counts = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=device,
            )

        input_span_counts = input_span_mask.sum(dim=1)
        if input_span_counts.sum() > 0 and patch_embeddings is None:
            raise RuntimeError(
                "Found audio span tokens but no latents provided to compute patch embeddings."
            )

        inputs_embeds = core.llm.get_input_embeddings()(input_ids)
        if patch_embeddings is not None:
            inputs_embeds = inputs_embeds.clone()
            patch_embeddings = patch_embeddings.to(inputs_embeds.dtype)
            for batch_idx in range(batch_size):
                span_num = int(input_span_counts[batch_idx].item())
                if span_num == 0:
                    continue
                expected = int(valid_patch_counts[batch_idx].item())
                if expected != span_num:
                    raise RuntimeError(
                        f"Mismatch between span tokens ({span_num}) and latent patches "
                        f"({expected}) for sample {batch_idx}."
                    )
                indices = input_span_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
                inputs_embeds[batch_idx, indices, :] = patch_embeddings[
                    batch_idx,
                    :span_num,
                    :,
                ]

        _llm_attn_mask, llm_seq_mask, _ = core.causal_helper.create_causal_mask_and_pos(
            seq_lens=input_ids_lengths,
            max_len=input_ids.size(1),
        )
        llm_outputs = core.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=llm_seq_mask.long(),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        llm_logits = llm_outputs.logits
        llm_hidden = llm_outputs.hidden_states[-1]
        eos = core.eos_proj(llm_hidden.detach())

        total_patches = int(output_span_mask.sum().item())
        if total_patches > 0 and latents_sampled is None:
            raise RuntimeError("MeanFlow training requested but latents are missing.")

        if total_patches > 0:
            pred, target = self.meanflow_fm_segment(
                data,
                llm_hidden=llm_hidden,
                inputs_embeds=inputs_embeds,
                output_span_mask=output_span_mask,
                latents_sampled=latents_sampled,
                latent_lengths=latent_lengths,
            )
        else:
            pred, target = self.dummy_fm_forward(core, llm_hidden, device)

        return DotsTtsForwardOutput(
            llm_logits=llm_logits,
            pred=pred,
            target=target,
            eos_out=eos,
        )

    def meanflow_fm_segment(
        self,
        data: dict[str, Any],
        *,
        llm_hidden: torch.Tensor,
        inputs_embeds: torch.Tensor,
        output_span_mask: torch.Tensor,
        latents_sampled: torch.Tensor,
        latent_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        core = self.student.core
        teacher_core = self.teacher.core
        settings = self.settings
        batch_size = latents_sampled.size(0)
        device = latents_sampled.device
        latent_dtype = latents_sampled.dtype
        first_t = torch.randn(batch_size, device=device, dtype=latent_dtype)
        second_t = torch.randn(batch_size, device=device, dtype=latent_dtype)
        first_t = torch.sigmoid(
            first_t * float(settings.time_sampling_std)
            + float(settings.time_sampling_mean)
        )
        second_t = torch.sigmoid(
            second_t * float(settings.time_sampling_std)
            + float(settings.time_sampling_mean)
        )
        t_vec = torch.minimum(first_t, second_t)
        delta_t = (first_t - second_t).abs()
        anchor_mask = torch.rand(batch_size, device=device, dtype=latent_dtype) < float(
            settings.anchor_prob
        )
        delta_t = torch.where(anchor_mask, torch.zeros_like(delta_t), delta_t)
        z0 = torch.randn_like(latents_sampled)
        xt = core.fm_helper.sample_x_t(
            z0,
            latents_sampled,
            t_vec.view(-1, 1, 1).to(latent_dtype),
        )

        fused_cfg = settings.cfg_distill_mode == "fused"
        if fused_cfg:
            cfg_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
            xvec_drop_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        else:
            cfg_mask = torch.empty(
                batch_size, device=device, dtype=torch.float32
            ).uniform_(0, 1) < float(core.cfg_droprate)
            xvec_drop_mask = torch.empty(
                batch_size, device=device, dtype=torch.float32
            ).uniform_(0, 1) < float(core.xvec_drop_rate)

        xvec_cond = core.xvec_proj(data["xvector"])
        vocal_mask = data.get("vocal_mask")
        if vocal_mask is None:
            vocal_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        xvec_cond = util_module.mask_data(xvec_cond, xvec_drop_mask & vocal_mask)

        hiddens_for_fm = torch.where(
            output_span_mask.unsqueeze(-1),
            llm_hidden,
            inputs_embeds,
        )
        prefix_data = core.io_helper.prepare_meanflow_inputs_for_dit(
            hiddens=hiddens_for_fm,
            latents=latents_sampled,
            latent_lens=latent_lengths,
            hidden_proj=core.hidden_proj,
            latent_proj=core.latent_proj,
            noisy_proj=core.coordinate_proj,
            span_mask=output_span_mask,
            hidden_patch_size=core.hidden_patch_size,
            latent_patch_size=core.latent_patch_size,
            cfg_mask=cfg_mask,
            noise_latents=xt,
        )

        uncond_prefix_data = None
        uncond_g_cond = None
        with torch.no_grad():
            teacher_xvec_cond = teacher_core.xvec_proj(data["xvector"])
            teacher_xvec_cond = util_module.mask_data(
                teacher_xvec_cond,
                xvec_drop_mask & vocal_mask,
            )
            teacher_prefix_data = (
                teacher_core.io_helper.prepare_meanflow_inputs_for_dit(
                    hiddens=hiddens_for_fm,
                    latents=latents_sampled,
                    latent_lens=latent_lengths,
                    hidden_proj=teacher_core.hidden_proj,
                    latent_proj=teacher_core.latent_proj,
                    noisy_proj=teacher_core.coordinate_proj,
                    span_mask=output_span_mask,
                    hidden_patch_size=teacher_core.hidden_patch_size,
                    latent_patch_size=teacher_core.latent_patch_size,
                    cfg_mask=cfg_mask,
                    noise_latents=xt,
                )
            )
            if fused_cfg:
                uncond_prefix_data = (
                    teacher_core.io_helper.prepare_meanflow_inputs_for_dit(
                        hiddens=hiddens_for_fm,
                        latents=latents_sampled,
                        latent_lens=latent_lengths,
                        hidden_proj=teacher_core.hidden_proj,
                        latent_proj=teacher_core.latent_proj,
                        noisy_proj=teacher_core.coordinate_proj,
                        span_mask=output_span_mask,
                        hidden_patch_size=teacher_core.hidden_patch_size,
                        latent_patch_size=teacher_core.latent_patch_size,
                        cfg_mask=torch.ones(
                            batch_size, device=device, dtype=torch.bool
                        ),
                        noise_latents=xt,
                    )
                )
                uncond_g_cond = torch.zeros_like(teacher_xvec_cond)

        teacher_target = self.compute_teacher_meanflow_target(
            xt=xt,
            t=t_vec,
            delta_t=delta_t,
            prefix_data=teacher_prefix_data,
            g_cond=teacher_xvec_cond,
            cfg_distill=fused_cfg,
            uncond_prefix_data=uncond_prefix_data,
            uncond_g_cond=uncond_g_cond,
        )
        if anchor_mask.any() and settings.anchor_target == "formula":
            target = self.replace_anchor_targets_with_formula(
                teacher_target,
                z0=z0,
                latents_sampled=latents_sampled,
                latent_lengths=latent_lengths,
                anchor_mask=anchor_mask,
            )
        else:
            target = teacher_target

        student_vt = core.velocity_field_predictor(
            x=prefix_data["fm_seq"],
            timesteps=t_vec,
            duration=delta_t,
            pos_ids=prefix_data["fm_pos_ids"],
            mask=prefix_data["fm_seq_mask"],
            attn_mask=prefix_data["fm_attn_mask"],
            g_cond=xvec_cond,
        )
        pred = core.io_helper.get_dit_outputs(
            pred_v=student_vt,
            fm_prefix_lengths=prefix_data["fm_prefix_lengths"],
            fm_gen_lengths=prefix_data["fm_gen_lengths"],
            fm_gen_patch_size=prefix_data["fm_gen_patch_size"],
            latent_patch_size=prefix_data["latent_patch_size"],
        )
        return pred, target

    def replace_anchor_targets_with_formula(
        self,
        teacher_target: torch.Tensor,
        *,
        z0: torch.Tensor,
        latents_sampled: torch.Tensor,
        latent_lengths: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        core = self.student.core
        formula_target = core.fm_helper.compute_u_t(z0, latents_sampled)
        chunks = []
        offset = 0
        for batch_idx in range(latents_sampled.size(0)):
            length = int(latent_lengths[batch_idx].item())
            if length <= 0:
                continue
            patch_count = length // core.latent_patch_size
            if bool(anchor_mask[batch_idx].item()):
                chunks.append(
                    rearrange(
                        formula_target[batch_idx, :length, :],
                        "(n p) d -> n p d",
                        p=core.latent_patch_size,
                    )
                )
            else:
                chunks.append(teacher_target[offset : offset + patch_count])
            offset += patch_count
        if not chunks:
            raise RuntimeError("Anchor target replacement produced no target.")
        return torch.cat(chunks, dim=0)

    def dummy_fm_forward(
        self,
        core,
        llm_hidden: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dummy_length = core.latent_patch_size
        dummy_seq_h = llm_hidden.new_zeros((1, dummy_length, core.llm_hidden_size))
        dummy_seq_h = core.hidden_proj(dummy_seq_h) * 0.0
        dummy_seq_l = llm_hidden.new_zeros((1, dummy_length, core.latent_dim))
        dummy_seq_l = core.latent_proj(dummy_seq_l) * 0.0
        dummy_seq_c = llm_hidden.new_zeros((1, dummy_length, core.latent_dim))
        dummy_seq_c = core.coordinate_proj(dummy_seq_c) * 0.0
        dummy_seq = dummy_seq_h + dummy_seq_l + dummy_seq_c
        dummy_times = torch.zeros((1,), device=device, dtype=torch.float32)
        dummy_duration = torch.zeros((1,), device=device, dtype=torch.float32)
        dummy_attn_mask = torch.ones(
            (1, dummy_length, dummy_length),
            device=device,
            dtype=torch.bool,
        )
        dummy_out = core.velocity_field_predictor(
            x=dummy_seq,
            timesteps=dummy_times,
            duration=dummy_duration,
            attn_mask=dummy_attn_mask,
        )
        pred = dummy_out[:, -core.latent_patch_size :, :]
        return pred, pred.detach()


class DotsTtsMeanFlowTrainingRun(DotsTtsTrainingRun):
    def __init__(
        self,
        cfg: app_config.AppConfig,
        *,
        meanflow_settings: MeanFlowSettings,
        debug_enabled: bool = False,
    ):
        self.cfg = cfg
        self.meanflow_settings = meanflow_settings
        self.progress = train_utils.TrainProgress()
        self.max_train_steps = int(cfg.train.max_train_steps)
        self.grad_accumulation_steps = int(cfg.train.gradient_accumulation_steps)
        self.last_validation_step: int | None = None
        self.consecutive_empty_epochs = 0
        self.saved_latest_checkpoint = False
        self._last_log_step = 0
        self._last_log_time = 0.0
        self._debug_enabled = bool(debug_enabled)
        self._debug_batch_count = 0
        self._debug_audio_sample_rate = int(self.cfg.train_data.train_audio_sample_rate)

        project_config = ProjectConfiguration(
            project_dir=self.cfg.train.output_dir,
            total_limit=self.cfg.train.max_checkpoints_to_keep,
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=self.grad_accumulation_steps,
            log_with="tensorboard",
            project_config=project_config,
            step_scheduler_with_optimizer=False,
        )

        util_module.seed_everything(self.cfg.train.seed)

        student = dots_tts_model.DotsTtsModel.from_pretrained(
            self.cfg.train.pretrained_model_path
        )
        student.set_cfg_droprate(
            cfg_droprate=self.cfg.train.cfg_droprate,
            xvec_drop_rate=self.cfg.train.xvec_drop_rate,
        )
        enable_meanflow_student(student)
        if not bool(meanflow_settings.train_all_parameters):
            for param in student.parameters():
                param.requires_grad_(False)
            for param in student.core.velocity_field_predictor.parameters():
                param.requires_grad_(True)
        model = MeanFlowDotsTtsModel(student, meanflow_settings)

        teacher_path = (
            meanflow_settings.teacher_model_path or self.cfg.train.pretrained_model_path
        )
        teacher = dots_tts_model.DotsTtsModel.from_pretrained(teacher_path)
        model.set_teacher(teacher)

        optimizer = AdamW(
            (param for param in model.parameters() if param.requires_grad),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.train.warmup_steps,
            num_training_steps=self.max_train_steps,
        )
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            model,
            optimizer,
            scheduler,
        )
        self.unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.unwrapped_model.to(self.accelerator.device)

        expected_sample_rate = int(self.unwrapped_model.config.vocoder.sample_rate)
        expected_audio_samples_per_llm_token = int(
            self.unwrapped_model.student.hop_size
        ) * int(self.unwrapped_model.config.patch_size)
        if int(self.cfg.train_data.train_audio_sample_rate) != expected_sample_rate:
            raise ValueError(
                f"train_data.train_audio_sample_rate={int(self.cfg.train_data.train_audio_sample_rate)} "
                f"does not match the pretrained model sample rate {expected_sample_rate}."
            )
        if (
            int(self.cfg.train_data.audio_samples_per_llm_token)
            != expected_audio_samples_per_llm_token
        ):
            raise ValueError(
                "train_data.audio_samples_per_llm_token="
                f"{int(self.cfg.train_data.audio_samples_per_llm_token)} "
                "does not match the pretrained model audio token contract "
                f"{expected_audio_samples_per_llm_token}."
            )
        if self.cfg.val_data is not None:
            if int(self.cfg.val_data.train_audio_sample_rate) != expected_sample_rate:
                raise ValueError(
                    f"val_data.train_audio_sample_rate={int(self.cfg.val_data.train_audio_sample_rate)} "
                    f"does not match the pretrained model sample rate {expected_sample_rate}."
                )
            if (
                int(self.cfg.val_data.audio_samples_per_llm_token)
                != expected_audio_samples_per_llm_token
            ):
                raise ValueError(
                    "val_data.audio_samples_per_llm_token="
                    f"{int(self.cfg.val_data.audio_samples_per_llm_token)} "
                    "does not match the pretrained model audio token contract "
                    f"{expected_audio_samples_per_llm_token}."
                )

        if self.accelerator.is_main_process:
            total_params = sum(
                param.numel() for param in self.unwrapped_model.parameters()
            )
            trainable_params = sum(
                param.numel()
                for param in self.unwrapped_model.parameters()
                if param.requires_grad
            )
            self.accelerator.print(f"Total parameters: {total_params:,}")
            self.accelerator.print(f"Trainable parameters: {trainable_params:,}")
            self.accelerator.print(
                f"MeanFlow teacher path: {Path(teacher_path).expanduser()}"
            )
            self.accelerator.print(
                f"Distributed type: {self.accelerator.distributed_type}"
            )

        tokenizer = self.unwrapped_model.tokenizer
        self.tokenizer = tokenizer
        train_dataset = data_module.build_training_dataset(
            self.cfg.train_data,
            tokenizer=tokenizer,
            seed=int(self.cfg.train.seed),
            accelerator=self.accelerator,
        )
        self.train_loader = data_module.build_training_dataloader(
            train_dataset,
            self.cfg.train_data,
            tokenizer=tokenizer,
        )

        self.val_loader = None
        if self.cfg.train.eval_interval is not None or self.cfg.train.run_eval_on_start:
            if self.cfg.val_data is None:
                raise ValueError(
                    "Validation requires val_data when eval_interval or "
                    "run_eval_on_start is enabled."
                )
            validation_data_cfg = self.cfg.val_data.model_copy(deep=True)
            validation_data_cfg.num_tokens_per_epoch = None
            val_dataset = data_module.build_validation_dataset(
                validation_data_cfg,
                tokenizer=tokenizer,
                seed=int(self.cfg.train.seed),
                accelerator=self.accelerator,
            )
            self.val_loader = data_module.build_validation_dataloader(
                val_dataset,
                validation_data_cfg,
                tokenizer=tokenizer,
            )

        self._resume_if_available()
        self.train_loader.set_epoch(self.progress.epoch)

    def _write_run_config(self) -> None:
        if not bool(getattr(self.accelerator, "is_main_process", True)):
            return
        output_dir = Path(self.cfg.train.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "config.yml"
        payload = self.cfg.to_dict()
        payload["meanflow_train"] = self.meanflow_settings.to_dict()
        with config_path.open("w", encoding="utf-8") as fout:
            yaml.safe_dump(
                payload,
                fout,
                sort_keys=False,
                allow_unicode=True,
            )

    def _save_checkpoint(self, learning_rate: float) -> None:
        train_checkpoint.save_train_checkpoint(
            self.accelerator,
            self.model,
            self.optimizer,
            self.progress,
            self.cfg.train.output_dir,
            self.cfg.train.max_checkpoints_to_keep,
            self.train_loader.state_dict(),
            {
                "type": "transformers_cosine_with_warmup_meanflow",
                "global_step": int(self.progress.global_step),
                "base_lr": float(self.cfg.train.learning_rate),
                "current_lr": float(learning_rate),
                "warmup_steps": int(self.cfg.train.warmup_steps),
                "max_train_steps": int(self.max_train_steps),
                "meanflow": self.meanflow_settings.to_dict(),
                "state_dict": self.scheduler.state_dict(),
            },
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Accelerate MeanFlow distillation entrypoint for dots.tts."
    )
    parser.add_argument("--config", default=app_config.DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print training debug information.",
    )
    parser.add_argument(
        "--teacher-model-path",
        default=None,
        help=(
            "Frozen flow-matching teacher model path. Defaults to "
            "train.pretrained_model_path."
        ),
    )
    parser.add_argument("--teacher-steps", type=int, default=8)
    parser.add_argument(
        "--teacher-solver",
        choices=_ALLOWED_TEACHER_SOLVERS,
        default="euler",
    )
    parser.add_argument(
        "--cfg-distill-mode",
        choices=_ALLOWED_CFG_DISTILL_MODES,
        default="fused",
    )
    parser.add_argument("--distill-cfg-scale", type=float, default=1.2)
    parser.add_argument("--anchor-prob", type=float, default=0.5)
    parser.add_argument(
        "--anchor-target",
        choices=_ALLOWED_ANCHOR_TARGETS,
        default="formula",
    )
    parser.add_argument("--time-sampling-mean", type=float, default=-0.4)
    parser.add_argument("--time-sampling-std", type=float, default=1.0)
    parser.add_argument(
        "--train-all-parameters",
        action="store_true",
        help="Train all regular dots.tts parameters instead of only the DiT.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    settings = MeanFlowSettings(
        teacher_model_path=args.teacher_model_path,
        teacher_steps=args.teacher_steps,
        teacher_solver=args.teacher_solver,
        cfg_distill_mode=args.cfg_distill_mode,
        distill_cfg_scale=args.distill_cfg_scale,
        anchor_prob=args.anchor_prob,
        anchor_target=args.anchor_target,
        time_sampling_mean=args.time_sampling_mean,
        time_sampling_std=args.time_sampling_std,
        train_all_parameters=args.train_all_parameters,
    )
    return DotsTtsMeanFlowTrainingRun(
        app_config.load_config(args.config),
        meanflow_settings=settings,
        debug_enabled=args.debug,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
