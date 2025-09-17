import tensorflow.keras.callbacks as callbacks
import time
import utils.SimpleTelebotReport as trep
from utils import DataPaths
import numpy as np
from matplotlib import pyplot as plt

class TrainingCallbacksData:
    def __init__(self, model, model_name, args, top1_float):
        self.model = model
        self.model_name = model_name
        self.args = args
        self.best_acc = 0.0
        self.best_acc_results = None
        self.best_acc_epoch = None
        self.epochs_counter = 1
        self.epoch_start = time.time()
        self.last_val_accs = []
        self.early_stopped = False
        self.top1_float = top1_float
        self.cur_loss:float = 99999.0
        self.last_lr_accs = []
        self.last_logs = []
        self.last_lrs = []
        self.epoch_events = {}
        self.cur_best_model_for_training_epoch = None

    @property
    def cur_epoch(self):
        return len(self.last_lrs)

    def add_event_at_cur_epoch(self, message):
        cur_epoch = self.cur_epoch
        if cur_epoch not in self.epoch_events:
            self.epoch_events[cur_epoch] = []
        self.epoch_events[cur_epoch].append(message)

class TrainingCallbacks(callbacks.Callback):

    def __init__(self, qTraining: TrainingCallbacksData, train_desc):
        super().__init__()
        self.qTraining = qTraining
        self.last_lr = None
        self.train_desc = train_desc
        pass

    def you_are_done(self):
        self.plot_and_report_progress(is_final=True)

    def plot_and_report_progress(self, is_final=False):
        model_base_path, MODEL_SAVEPATH, MODEL_SAVEPATH_Q, MODEL_SAVEPATH_BEST, MODEL_SAVEPATH_Q_BEST, MODEL_SAVEPATH_Q_FUSEB = DataPaths.get_model_paths(self.qTraining.args)

        def get_all(name):
            result = []
            for log_entry in self.qTraining.last_logs:
                if name in log_entry:
                    result.append(log_entry[name])
            return result


        all_train_loss = get_all("loss")
        all_val_loss = get_all("val_loss")
        all_train_top1 = get_all("top 1")
        all_val_top1 = get_all("val_top 1")
        epochs = [i+1 for i, _ in enumerate(all_train_loss)]
        all_lrs = self.qTraining.last_lrs

        best_val_top1_epoch = epochs[np.argmax(all_val_top1)]
        best_val_top1 = np.max(all_val_top1)

        fig, ax1 = plt.subplots(figsize=(25, 7))

        # Plot accuracy (left y-axis)
        train_acc_line, = ax1.plot(epochs, all_train_top1, '--', color="skyblue", label='train top1')
        val_acc_line, = ax1.plot(epochs, all_val_top1, '-', color="blue", label='val top1')
        ax1.scatter(best_val_top1_epoch, best_val_top1, color='blue', marker='*', s=200, label='best val top1')
        best_val_top1_prozent = round(best_val_top1 * 100, 2)
        best_val_top1_prozent = f'{best_val_top1_prozent:.2f}%'
        ax1.text(best_val_top1_epoch, best_val_top1, best_val_top1_prozent, color='b', fontsize=12, ha='right', va='bottom')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        if len(epochs) < 100:
            ax1.set_xticks(epochs)
            ax1.set_xticklabels(epochs, rotation=40)
        elif len(epochs) < 200:
            ax1.set_xticks([x for x in epochs if x % 2 == 0])
            ax1.set_xticklabels([x for x in epochs if x % 2 == 0], rotation=40)
        else:
            ax1.set_xticks([x for x in epochs if x % 5 == 0])
            ax1.set_xticklabels([x for x in epochs if x % 5 == 0], rotation=40)

        # Double the number of y-ticks for the first axis
        original_ticks = ax1.get_yticks()
        new_ticks = np.linspace(original_ticks[0], original_ticks[-1], len(original_ticks) * 2)  # Double tick count
        ax1.set_yticks(new_ticks)
        ax1.set_yticklabels([f'{y * 100:.2f}%'  for y in new_ticks])


        # Create second y-axis for loss
        ax2 = ax1.twinx()
        train_loss_line, = ax2.plot(epochs, all_train_loss, '--', color="lightsalmon", label='train loss')
        val_loss_line, = ax2.plot(epochs, all_val_loss, '-', color="red", label='val loss')
        ax2.set_ylabel('Loss', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Create third y-axis for learning rate
        ax3 = ax1.twinx()
        lr_line, = ax3.plot(epochs, all_lrs, 'g:', label='Learning Rate')
        ax3.set_ylabel('Learning Rate', color='g')
        ax3.tick_params(axis='y', labelcolor='g')

        # Offset third axis to avoid overlap
        ax3.spines["right"].set_position(("outward", 60))

        # Add horizontal lines at tick positions for the first axis (accuracy)
        for ytick in new_ticks:
            ax1.axhline(ytick, color='gray', linestyle='dashed', linewidth=0.5, alpha=0.5)

        # Print event messages at the correct epoch positions
        for epoch, messages in self.qTraining.epoch_events.items():
            msg_text = "\n".join(messages)  # Combine multiple messages
            text = ax1.text(epoch, all_train_top1[epoch-1], msg_text, color='black', fontsize=8, ha='center',
                     bbox=dict(facecolor="red", alpha=0.4, edgecolor=None))
            text.set_bbox(dict(facecolor='red', alpha=0.0, edgecolor='red'))

        # Collect handles and labels for the legend
        handles = [train_acc_line, val_acc_line, train_loss_line, val_loss_line, lr_line]
        labels = [h.get_label() for h in handles]

        # Add legend outside the plot
        plt.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=5)

        # Show the plot
        plt.title(self.train_desc + f" (best val top1 {best_val_top1_prozent})")
        fig.tight_layout()
        report_path = model_base_path.format("last_train_report.png")
        plt.savefig(report_path)
        message = self.train_desc
        if is_final:
            message = "FINAL REPORT: " + message
        trep.sendImageReport(message, report_path)
        if is_final:
            time.sleep(5)
            trep.sendModelReport(self.qTraining.model_name, self.qTraining.args, "Done")
            time.sleep(5)
            trep.sendReport("Saved under path " + model_base_path)
        plt.close(fig)


    def on_epoch_end(self, epoch, logs={}):
        model_base_path, MODEL_SAVEPATH, MODEL_SAVEPATH_Q, MODEL_SAVEPATH_BEST, MODEL_SAVEPATH_Q_BEST, MODEL_SAVEPATH_Q_FUSEB = DataPaths.get_model_paths(self.qTraining.args)

        cur_acc = logs.get("val_fixed_(strict)_top_1",
                           logs.get("val_fixed (strict) top 1"))
        cur_loss = logs["val_loss"]
        self.qTraining.last_logs.append(logs)
        cur_lr = float(np.array(self.model.optimizer.lr))
        self.qTraining.last_lrs.append(cur_lr)

        if (len(self.qTraining.last_lrs) % 5 == 0 and len(self.qTraining.last_lrs) > 1) or len(self.qTraining.last_lrs) == 5:
            self.plot_and_report_progress()

        try:
            cur_lr = float(np.array(self.model.optimizer.lr))
            if self.last_lr is None:
                self.last_lr = cur_lr
            elif self.last_lr != cur_lr:
                print(epoch, self.last_lr, "->", cur_lr, self.last_lr_accs)
                # trep.sendModelReport(self.qTraining.model_name, self.qTraining.args, f"Changing LR on epoch {epoch} from {self.last_lr} -> {cur_lr}. val top 1 since last LR change were: {self.qTraining.last_lr_accs}")
                self.last_lr = cur_lr
                self.qTraining.last_lr_accs = []
            else:
                self.qTraining.last_lr_accs.append(cur_acc)
        except:
            pass

        if "nan" in str(cur_loss).lower() or "inf" in str(cur_loss).lower():
            cur_loss = 9999.9
        self.qTraining.last_val_accs.append(cur_acc)
        # if len(self.qTraining.last_val_accs) == 1 or len(self.qTraining.last_val_accs) == 3 or len(
        #         self.qTraining.last_val_accs) == 5 or len(self.qTraining.last_val_accs) == 15:
        #     trep.sendModelReport(self.qTraining.model_name, self.qTraining.args,
        #                          f"Last {len(self.qTraining.last_val_accs)} epochs are {self.qTraining.last_val_accs} fp32 would be {self.qTraining.top1_float}")
        with open(model_base_path.format("epochs_log.txt"), "a") as f:
            history_entry = f"{self.qTraining.epochs_counter} | {epoch} | {self.qTraining.model.optimizer.lr.numpy()} | {str(logs)} | {self.qTraining.best_acc < cur_acc} | {time.time() - self.qTraining.epoch_start}"
            f.write(history_entry + "\n")
        self.qTraining.epochs_counter += 1
        self.qTraining.epoch_start = time.time()
        if self.qTraining.best_acc < cur_acc:
            self.qTraining.best_acc = cur_acc
            self.qTraining.best_acc_results = logs
            self.qTraining.best_acc_epoch = self.qTraining.epochs_counter
            self.model.saveVariablesNPZ(MODEL_SAVEPATH_BEST, quantisize=False)
            self.model.saveVariablesNPZ(MODEL_SAVEPATH_Q_BEST, quantisize=True)
            self.qTraining.add_event_at_cur_epoch("ValBest")
        if self.qTraining.cur_loss > cur_loss:
            print()
            print(f"Found better one! {self.qTraining.cur_loss} -> {cur_loss} ")
            if cur_acc >= self.qTraining.top1_float:
                self.model.stop_training = True
                self.qTraining.early_stopped = True
                print(f"EARLY STOPPING! Already at float acc!")
            self.qTraining.cur_loss = cur_loss
            with open(model_base_path.format("results.txt"), "w") as f:
                f.write(str(logs))
            with open(model_base_path.format("config.json"), "w") as f:
                f.write(str(self.qTraining.args))
            self.qTraining.model.saveVariablesNPZ(MODEL_SAVEPATH, quantisize=False)
            self.qTraining.model.saveVariablesNPZ(MODEL_SAVEPATH_Q, quantisize=True)
            self.qTraining.cur_best_model_for_training_epoch = self.qTraining.cur_epoch
            self.qTraining.add_event_at_cur_epoch("TrnBest")
