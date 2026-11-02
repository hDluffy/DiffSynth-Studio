import torch


try:
    import deepspeed
    _HAS_DEEPSPEED = True
except ModuleNotFoundError:
    _HAS_DEEPSPEED = False


def create_custom_forward(module):
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)
    return custom_forward


def create_custom_forward_use_reentrant(module):
    def custom_forward(*inputs):
        return module(*inputs)
    return custom_forward


def judge_args_requires_grad(*args):
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.requires_grad:
            return True
    return False


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    *args,
    checkpoint_use_reentrant=False,
    **kwargs,
):
    if use_gradient_checkpointing and _HAS_DEEPSPEED and deepspeed.checkpointing.is_configured():
        all_args = args + tuple(kwargs.values())
        if not judge_args_requires_grad(*all_args):
            # get the first grad_enabled tensor from un_checkpointed forward
            model_output = model(*args, **kwargs)
        else:
            model_output = deepspeed.checkpointing.checkpoint(
                create_custom_forward_use_reentrant(model),
                *all_args,
            )
        return model_output
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            if checkpoint_use_reentrant:
                # Reentrant checkpointing runs the first forward under no_grad,
                # so save_on_cpu only stores the explicit checkpoint inputs
                # instead of every tensor saved by operations inside the block.
                def custom_forward(*inputs):
                    return model(*inputs, **kwargs)

                checkpoint_args = args
                if torch.is_grad_enabled() and not judge_args_requires_grad(*args):
                    # Frozen inputs are common in LoRA training. Reentrant
                    # checkpointing needs a grad-enabled explicit input to
                    # retain gradients for trainable parameters inside model.
                    dummy = torch.ones((), requires_grad=True)
                    checkpoint_args = (dummy,) + args

                    def custom_forward(_dummy, *inputs):
                        return model(*inputs, **kwargs)

                model_output = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    *checkpoint_args,
                    use_reentrant=True,
                )
            else:
                model_output = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(model),
                    *args,
                    **kwargs,
                    use_reentrant=False,
                )
    elif use_gradient_checkpointing:
        model_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
        )
    else:
        model_output = model(*args, **kwargs)
    return model_output
