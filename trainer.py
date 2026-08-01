import torch
import torch.nn as nn
import numpy as np
import os

from record import PerformanceMeter
from utils import count_parameters

class Trainer(nn.Module):
    r'''A Multi-Task Learning Trainer.
    Assumes data from DataLoader is already collated (by DataCollator as collate_fn)
    and is a dictionary of tensors.
    Model's forward pass is now expected to return a single value:
    - A dictionary of predictions for 'train' (final step) and 'val'/'test' modes.
    - None for 'train' mode intermediate steps (encoder accumulating).
    '''

    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders,
                 optim_param, args, save_path = None, load_path = None, **kwargs):
        super(Trainer, self).__init__()

        if args.gpu_id != 'cpu' and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{args.gpu_id}')
        else:
            self.device = torch.device('cpu')

        self.kwargs = kwargs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())  # Trainer's list of tasks
        self.save_path = save_path
        self.load_path = load_path
        self.args = args

        self.prepare_model(weighting, architecture, encoder_class, decoders)
        self.prepare_optimizer(optim_param)
        self.meter = PerformanceMeter(self.task_dict)

    def prepare_model(self, weighting, architecture, encoder_class, decoders):
        class MTLmodel(architecture, weighting):
            def __init__(self, task_name_list, enc_class, dec_dict, dev, cmd_args, arch_kwargs_val):
                super(MTLmodel, self).__init__(task_name_list, enc_class, dec_dict, dev, cmd_args, **arch_kwargs_val)
                if hasattr(self, 'init_param'):
                    self.init_param()
                else:
                    print(
                        f"DEBUG: Model class {type(self).__name__} (bases: {type(self).__bases__}) does not have an init_param method.")

        self.model = MTLmodel(task_name_list = self.task_name,
                              enc_class = encoder_class,
                              dec_dict = decoders,
                              dev = self.device,
                              cmd_args = self.args,
                              arch_kwargs_val = self.kwargs.get('arch_args', {})
                              ).to(self.device)

        # DEBUG prints (can be commented out after verification)
        # print(f"DEBUG: self.model type: {type(self.model)}")
        # print(f"DEBUG: self.model MRO: {type(self.model).mro()}")
        # print(f"DEBUG: self.model.forward bound method: {self.model.forward}")

        if self.load_path is not None:
            if os.path.isdir(self.load_path):
                ckpt_file_name = f'{self.args.ckpt_name}_best.pt' if hasattr(self.args, 'ckpt_name') else 'best.pt'
                self.load_path = os.path.join(self.load_path, ckpt_file_name)

            if os.path.exists(self.load_path):
                try:
                    self.model.load_state_dict(torch.load(self.load_path, map_location = self.device), strict = False)
                    print('Successfully loaded model from - {}'.format(self.load_path))
                except Exception as e:
                    print(
                        f"Error loading model from {self.load_path}: {e}. Training from scratch or with initial weights.")
            else:
                print(f"Model file not found at {self.load_path}. Training from scratch or with initial weights.")
        count_parameters(self.model)

    def prepare_optimizer(self, optim_param):
        optim_dict = {'adam': torch.optim.Adam}
        optim_name = optim_param.get('optim', 'adam').lower()
        if optim_name not in optim_dict:
            raise ValueError(f"Unsupported optimizer: {optim_name}")
        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_name](self.model.parameters(), **optim_arg)

    def compute_loss(self, preds, gts, task_name = None):
        loss = self.meter.losses[task_name].update_loss(preds, gts.to(preds.device))
        return loss

    def _prepare_iterators_and_counts(self, dataloaders_dict):
        iterators = {}
        counts = {}
        if dataloaders_dict:
            for task, loader in dataloaders_dict.items():
                if loader:  # Check if loader itself is not None
                    if len(loader) > 0:
                        iterators[task] = iter(loader)
                        counts[task] = len(loader)
                    else:  # Loader exists but is empty
                        iterators[task] = None  # No iterator for empty loader
                        counts[task] = 0
                        print(f"Warning: DataLoader for task '{task}' is empty (length 0).")
                else:  # Loader is None
                    counts[task] = 0
                    iterators[task] = None
                    print(f"Warning: DataLoader for task '{task}' is None.")
        return iterators, counts

    def train(self, train_dataloaders_dict, val_dataloaders_dict, test_dataloaders_dict, epochs, params_main):
        train_iterators, train_batch_counts = self._prepare_iterators_and_counts(train_dataloaders_dict)

        # Determine max_batches_per_epoch based on tasks that actually have data
        active_train_batch_counts = [count for task, count in train_batch_counts.items() if
                                     train_iterators.get(task) is not None]
        if not active_train_batch_counts:
            print("Error: All training dataloaders are effectively empty. Cannot start training.")
            return
        max_batches_per_epoch = max(active_train_batch_counts) if active_train_batch_counts else 0
        if max_batches_per_epoch == 0:
            print("Error: max_batches_per_epoch is 0. All tasks might have empty dataloaders. Cannot start training.")
            return

        if hasattr(self.model, 'train_loss_buffer'): self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        if hasattr(self.model, 'epochs'): self.model.epochs = epochs
        self.batch_weight = np.zeros([self.task_num, epochs, max_batches_per_epoch])

        for epoch in range(epochs):
            if hasattr(self.model, 'epoch'): self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')

            for batch_idx in range(max_batches_per_epoch):
                current_epoch_batch_step_losses = torch.zeros(self.task_num, device = self.device)
                ground_truths_for_step = {}
                active_tasks_in_step = []  # Tasks that successfully provided data for this batch_idx
                model_output_predictions_final_for_step = None

                for task_order_index, current_task_name in enumerate(self.task_name):
                    batch_data_for_task = None
                    current_task_iterator = train_iterators.get(current_task_name)

                    if current_task_iterator is None:  # Task had an empty dataloader initially
                        # print(f"DEBUG: Skipping task {current_task_name} for batch_idx {batch_idx} as its dataloader is empty/None.")
                        # For the Encoder to work, it needs an input for each of its configured tasks.
                        # If a task is truly empty, the Encoder needs to be robust to this.
                        # The current Encoder design appends to stack_task. If a task is skipped,
                        # len(stack_task) will be less than len(encoder.task_name), causing issues.
                        # This implies tasks in self.task_name given to Trainer MUST have non-empty dataloaders.
                        # The following logic ensures an attempt to get data, re-iterating if needed.
                        # If a task's dataloader was initially empty (len=0), this won't help.
                        # The check `if not active_train_batch_counts:` above should catch if ALL are empty.
                        # If *some* are empty, this loop structure needs the Encoder to handle missing appends,
                        # or these tasks should be excluded from self.task_name passed to Encoder/Trainer.
                        # For now, we proceed assuming tasks in self.task_name passed to Trainer *should* have data.
                        # If train_iterators[current_task_name] is None, it means len(loader) was 0.
                        # We cannot fetch data from it.
                        if epoch == 0 and batch_idx == 0:  # Print warning once
                            print(
                                f"Warning: Task {current_task_name} has an empty DataLoader. It will be skipped in model input feeding.")
                        continue  # Skip this task for model input if its loader is fundamentally empty.

                    try:
                        batch_data_for_task = next(current_task_iterator)
                    except StopIteration:
                        # Data exhausted for this task in the current epoch pass, re-initialize
                        if not train_dataloaders_dict.get(current_task_name) or \
                                len(train_dataloaders_dict[current_task_name]) == 0:
                            # This should have been caught by current_task_iterator being None
                            print(
                                f"ERROR: Task {current_task_name} DataLoader is unexpectedly None or empty upon StopIteration.")
                            continue

                        if epoch == 0 and batch_idx < 2:  # Print only for first few batches of first epoch for brevity
                            print(
                                f"INFO: Re-initializing iterator for task {current_task_name} at epoch {epoch}, batch {batch_idx}")

                        train_iterators[current_task_name] = iter(train_dataloaders_dict[current_task_name])
                        try:
                            batch_data_for_task = next(train_iterators[current_task_name])
                        except StopIteration:
                            print(f"ERROR: Task {current_task_name} failed to yield data even after re-initializing. "
                                  f"Its dataset might be truly empty or too small for batch_size {self.args.bs}. Skipping this task for this step.")
                            continue  # Skip if still no data

                    # We now have batch_data_for_task (or skipped if error above)
                    if batch_data_for_task is None:  # Should be caught by error handling above
                        continue

                    if batch_data_for_task.get('is_empty', False):
                        # This means the custom collator received an empty list or filtered all items.
                        # This can happen if dataset is very small and drop_last=True, and this is the last partial batch.
                        # Or if PreprocessedDatasetWrapper returned many Nones.
                        # For the Encoder to complete its cycle, it needs input.
                        # What to do here is tricky. Skipping might break Encoder's stack.
                        # Forcing a dummy input would be complex.
                        if epoch == 0 and batch_idx == 0:  # Print warning once
                            print(
                                f"Warning: Collator returned an empty batch for task {current_task_name} at epoch {epoch}, batch {batch_idx}. Skipping model call for this task's data.")
                        # If we skip, and this is a task the Encoder needs for its sequence, it can be problematic.
                        # However, if a batch is truly empty, there's no data to process.
                        # The Encoder's logic for handling len(stack_task) != len(encoder.task_name) would be critical.
                        # The fix in Encoder to reset stack_task at task_index==0 is key.
                        # If an intermediate task provides an empty batch, atoms_embedder_out for it might be problematic
                        # or `self.stack_task.append(atoms_embedder_out)` would get a problematic tensor.
                        # Let's assume atoms_emb can handle this or collator doesn't make it empty often.
                        # For now, if batch is empty, we cannot reliably feed it to the model as is.
                        continue

                    ground_truths_for_step[current_task_name] = batch_data_for_task['y']
                    if current_task_name not in active_tasks_in_step:
                        active_tasks_in_step.append(current_task_name)

                    # Model now returns one value: preds dict or None
                    temp_preds = self.model(batch_data_for_task, current_task_name, 'train')

                    if temp_preds is not None:  # Encoder finished its cycle
                        model_output_predictions_final_for_step = temp_preds

                # After iterating all tasks for model input accumulation for this batch_idx:
                if model_output_predictions_final_for_step is not None:
                    losses_for_backward_pass = []
                    for loss_calc_idx, task_name_for_loss in enumerate(self.task_name):
                        # Only calculate loss if task was active, has prediction, and has GT
                        if task_name_for_loss in active_tasks_in_step and \
                                task_name_for_loss in model_output_predictions_final_for_step and \
                                task_name_for_loss in ground_truths_for_step:
                            pred_tensor = model_output_predictions_final_for_step[task_name_for_loss]
                            gt_tensor = ground_truths_for_step[task_name_for_loss]

                            loss_val = self.compute_loss(pred_tensor, gt_tensor, task_name_for_loss)
                            current_epoch_batch_step_losses[loss_calc_idx] = loss_val
                            losses_for_backward_pass.append(loss_val)
                            self.meter.update(pred_tensor, gt_tensor, task_name_for_loss)  # Updates metrics
                        # else: current_epoch_batch_step_losses[loss_calc_idx] remains 0.0 if task was not active or had no prediction

                    if not losses_for_backward_pass:  # No valid losses to backprop
                        if batch_idx == 0 and epoch == 0 and active_tasks_in_step:  # Print only once for brevity
                            print(f"Warning: Train Batch {batch_idx}: Active tasks were {active_tasks_in_step}, "
                                  f"but no losses computed. Predictions: "
                                  f"{model_output_predictions_final_for_step.keys() if model_output_predictions_final_for_step else 'None'}. "
                                  f"Ground truths for: {ground_truths_for_step.keys()}. Skipping backward pass.")
                        continue

                    self.optimizer.zero_grad()
                    output_weights = self.model.backward(current_epoch_batch_step_losses,
                                                         **self.kwargs.get('weight_args', {}))
                    if output_weights is not None:
                        self.batch_weight[:, epoch, batch_idx] = output_weights.cpu().numpy() if isinstance(
                            output_weights, torch.Tensor) else output_weights
                    self.optimizer.step()

                elif active_tasks_in_step:  # Data was fed, but model didn't produce final output this step
                    if epoch == 0 and batch_idx == 0:  # Print warning only once for brevity
                        print(
                            f"Warning: Train Batch {batch_idx}: Model processed inputs from active tasks ({active_tasks_in_step}) "
                            f"but did not return final predictions. This is expected if the encoder's full task sequence "
                            f"for this meta-batch was not completed (e.g. if last task in encoder sequence had no data this batch_idx).")

            # --- End of epoch ---
            self.meter.record_time('end')
            self.meter.get_score()
            if hasattr(self.model, 'train_loss_buffer'): self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(mode = 'train', epoch = epoch)
            self.meter.reinit()

            if val_dataloaders_dict and any(val_dataloaders_dict.values()): self.test(val_dataloaders_dict,
                                                                                      epoch = epoch, mode = 'val')
            if test_dataloaders_dict and any(test_dataloaders_dict.values()): self.test(test_dataloaders_dict,
                                                                                        epoch = epoch, mode = 'test')

            if self.save_path is not None:
                is_best_epoch = False
                # Assuming PerformanceMeter's best_val_epoch['average'][0] holds the epoch number of the best average validation metric
                avg_best_epoch_info = self.meter.best_val_epoch.get('average')
                if avg_best_epoch_info and avg_best_epoch_info[0] == epoch:
                    is_best_epoch = True

                if is_best_epoch:
                    model_save_file = os.path.join(self.save_path, f'{params_main.ckpt_name}_best.pt')
                    torch.save(self.model.state_dict(), model_save_file)
                    print(f'Saved Best Model (epoch {epoch}) based on validation to {model_save_file}')

        self.meter.display_best_result()

    def test(self, dataloaders_dict, epoch = None, mode = 'test'):
        test_iterators, test_batch_counts = self._prepare_iterators_and_counts(dataloaders_dict)
        if not any(test_batch_counts.values()):
            print(f"No data for {mode} mode. Skipping.")
            return

        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            for task_order_index, task_name in enumerate(self.task_name):
                current_task_iterator = test_iterators.get(task_name)
                current_task_batch_count = test_batch_counts.get(task_name, 0)

                if not current_task_iterator or current_task_batch_count == 0:
                    # print(f"DEBUG: Skipping task {task_name} in {mode} mode as it has no data.")
                    continue

                if hasattr(self.meter, 'cache_result') and task_name in self.meter.cache_result:
                    self.meter.cache_result[task_name]['pred'] = []
                    self.meter.cache_result[task_name]['gts'] = []

                task_processed_at_least_one_batch = False
                for batch_idx in range(current_task_batch_count):
                    try:
                        batch_data = next(
                            current_task_iterator)  # In test mode, we don't re-initialize, just iterate through once
                        if batch_data.get('is_empty', False): continue

                        predictions_output = self.model(batch_data, task_name, mode)  # Model returns dict for this task

                        if predictions_output is not None and task_name in predictions_output:
                            current_task_pred = predictions_output[task_name]
                            current_task_gt = batch_data['y']
                            self.meter.append_result(current_task_pred, current_task_gt, task_name)
                            task_processed_at_least_one_batch = True
                        else:
                            print(
                                f"Warning: No prediction output for task {task_name} in {mode} mode, batch {batch_idx}.")
                    except StopIteration:  # Should ideally not happen if looping up to test_batch_counts
                        print(f"Warning: StopIteration unexpected in {mode} for task {task_name} at batch {batch_idx}.")
                        break
                    except Exception as e:
                        print(f"Error during {mode} for task {task_name}, batch {batch_idx}: {e}")
                        continue

                if task_processed_at_least_one_batch and \
                        hasattr(self.meter, 'cache_result') and \
                        self.meter.cache_result.get(task_name) and \
                        self.meter.cache_result[task_name]['pred'] and \
                        self.meter.cache_result[task_name]['gts']:

                    pred_all_for_task = torch.cat(self.meter.cache_result[task_name]['pred'], dim = 0)
                    gts_all_for_task = torch.cat(self.meter.cache_result[task_name]['gts'], dim = 0)

                    self.compute_loss(pred_all_for_task, gts_all_for_task, task_name)  # Record loss for val/test
                    self.meter.update(pred_all_for_task, gts_all_for_task, task_name)  # Update metrics
                elif not task_processed_at_least_one_batch and current_task_batch_count > 0:
                    print(
                        f"Warning: No batches successfully processed for task {task_name} in {mode} mode, though loader had data.")

        self.meter.record_time('end')
        self.meter.get_score()
        self.meter.display(mode = mode, epoch = epoch)
        self.meter.reinit()
            # # 1. 提取 EMA prompts
            # # shape [1, num_tasks, hidden_dim]
            # prompts = self.model.encoder.ema_prompts.squeeze(0).cpu().numpy()
            #
            # # 2. 将任务映射到物种
            # species_list = self.args.species_list
            # if not species_list:
            #     return
            #
            # def get_species_from_task(task_name, species_list):
            #     for species in species_list:
            #         if task_name.startswith(species.replace(" ", "_")):  # 替换空格以匹配任务名
            #             return species
            #     return 'unknown'
            #
            # task_labels = self.task_name
            # species_labels = [get_species_from_task(t, species_list) for t in task_labels]
            #
            # # 3. 执行 t-SNE 降维
            # # Perplexity 建议小于样本数。任务数通常不多，设一个较小的值。
            # perplexity_val = min(35, len(task_labels) - 1)
            # tsne = TSNE(n_components = 2, verbose = 1, perplexity = perplexity_val, n_iter = 1000, random_state = 42)
            # tsne_results = tsne.fit_transform(prompts)
            #
            # # 4. 使用 Seaborn 和 Matplotlib 绘图
            # df = pd.DataFrame()
            # df['tsne-1'] = tsne_results[:, 0]
            # df['tsne-2'] = tsne_results[:, 1]
            # df['species'] = species_labels
            # df['task'] = task_labels
            #
            #
            # def shorten_label(label):
            #     # 示例缩写规则，您可以根据您的任务名称格式自定义
            #     parts = label.replace('-', '_').split('_')
            #     if len(parts) >= 3:
            #         # 例如 'mouse_intraperitoneal_LD50' -> 'mouse_ip_LD50'
            #         route = parts[1]
            #         if route == 'intraperitoneal': route = 'ip'
            #         if route == 'intravenous': route = 'iv'
            #         if route == 'oral': route = 'o'
            #         if route == 'subcutaneous': route = 'sc'
            #         return f"{parts[0]}_{route}_{parts[-1]}"
            #     return label  # 如果不匹配规则，返回原标签
            #
            # df['short_task'] = df['task'].apply(shorten_label)
            # plt.figure(figsize = (16, 8), dpi = 300)
            # ax = sns.scatterplot(
            #     x = "tsne-1", y = "tsne-2",
            #     hue = "species",
            #     palette = sns.color_palette("hls", len(df['species'].unique())),
            #     data = df,
            #     legend = "full",
            #     alpha = 0.9,
            #     s = 200  # 点的大小
            # )
            #
            # # 为每个点添加任务名称注释
            # # for i in range(df.shape[0]):
            # #     plt.text(x = df['tsne-1'][i] + 0.1, y = df['tsne-2'][i] + 0.1, s = df['task'][i],
            # #              fontdict = dict(color = 'black', size = 8))
            # texts = []
            # for i in range(df.shape[0]):
            #     texts.append(plt.text(df['tsne-1'][i], df['tsne-2'][i], df['short_task'][i], size = 10))
            # # a-SNE 智能调整文本位置
            # # expand_points 增加点周围的填充
            # # arrowprops 设置从文本指向点的箭头样式
            # adjust_text(texts,
            #             ax = ax,
            #             expand_points = (1.2, 1.2),
            #             arrowprops = dict(arrowstyle = "-", color = 'gray', lw = 0.5))
            #
            # plt.xlabel('t-SNE Dimension 1')
            # plt.ylabel('t-SNE Dimension 2')
            #
            # # 保存图像
            # output_filename = 'task_prompts_tsne.png'
            # if self.save_path:
            #     output_filename = os.path.join(self.save_path, output_filename)
            #
            # plt.savefig(output_filename)
            # plt.close()
