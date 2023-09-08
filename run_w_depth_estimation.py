

def train_one_epoch(
        lic_model,
        de_model,
        train_dataloader,
        test_dataloader,
        metrics_tracker,
        context,
        config,
        run_id=None,
        save_dir=None,
        step=0,
):



def main(config: Dict[str, Any], args: argparse.Namespace):
    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    model = IDisc.build(config)
    model.load_pretrained(args.model_file)
    model = model.to(device)
    model.eval()

    f16 = config["training"].get("f16", False)
    context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=f16)

    save_dir = os.path.join(args.base_path, config["data"]["data_root"])
    assert hasattr(
        custom_dataset, config["data"]["train_dataset"]
    ), f"{config['data']['train_dataset']} not a custom dataset"
    valid_dataset = getattr(custom_dataset, config["data"]["val_dataset"])(
        test_mode=True, base_path=save_dir, crop=config["data"]["crop"]
    )
    valid_sampler = SequentialSampler(valid_dataset)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=4,
        sampler=valid_sampler,
        pin_memory=True,
        drop_last=False,
    )

    is_normals = config["model"]["output_dim"] > 1
    if is_normals:
        metrics_tracker = RunningMetric(list(DICT_METRICS_NORMALS.keys()))
    else:
        metrics_tracker = RunningMetric(list(DICT_METRICS_DEPTH.keys()))

    print("Start validation...")
    with torch.no_grad():
        validate.best_loss = np.inf
        validate(
            model,
            test_loader=valid_loader,
            config=config,
            metrics_tracker=metrics_tracker,
            context=context,
        )


def train_one_epoch(
    model, criterion, train_dataloader, test_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    model.update(force=True)
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device) # [ 16, 3, 256, 256 ]

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d) # [ 16, 3, 256, 256 ]

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def validate(
    model: nn.Module,
    test_loader: DataLoader,
    metrics_tracker: RunningMetric,
    context: torch.autocast,
    config: Dict[str, Any],
    run_id: Optional[str] = None,
    save_dir: Optional[str] = None,
    step: int = 0,
):
    ds_losses = {}
    device = model.device
    if save_dir is not None:
        run_save_dir = os.path.join(save_dir, run_id)
        os.makedirs(run_save_dir, exist_ok=True)

    for i, batch in enumerate(test_loader):
        with context:
            gt, mask = batch["gt"].to(device), batch["mask"].to(device)
            preds, losses, _ = model(batch["image"].to(device), gt, mask)

        losses = {k: v for l in losses.values() for k, v in l.items()}
        for loss_name, loss_val in losses.items():
            ds_losses[loss_name] = (
                loss_val.detach().cpu().item() + i * ds_losses.get(loss_name, 0.0)
            ) / (i + 1)

        metrics_tracker.accumulate_metrics(
            gt.permute(0, 2, 3, 1), preds.permute(0, 2, 3, 1), mask.permute(0, 2, 3, 1)
        )

    losses_all = ds_losses
    metrics_all = metrics_tracker.get_metrics()
    metrics_tracker.reset_metrics()

    if is_main_process():
        log_losses(losses_all=losses_all)
        update_best(metrics_all=metrics_all, metrics_best="abs_rel")
        if save_dir is not None:
            with open(os.path.join(run_save_dir, f"metrics_{step}.json"), "w") as f:
                json.dump({**losses_all, **metrics_all}, f)
            save_model(
                metrics_all=metrics_all,
                state_dict=model.state_dict(),
                config=config,
                metrics_best="abs_rel",
                run_save_dir=run_save_dir,
                step=step,
            )