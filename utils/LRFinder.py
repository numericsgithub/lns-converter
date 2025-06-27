import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class LRFinder(Callback):
    """Callback that exponentially adjusts the learning rate after each training batch between start_lr and
    end_lr for a maximum number of batches: max_step. The loss and learning rate are recorded at each step allowing
    visually finding a good learning rate as per https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html via
    the plot method.
    """

    def __init__(self, run_id, npz_filepath, png_filepath, start_lr: float = 1e-7, end_lr: float = 0.0004, max_steps: int = 4000, smoothing=0.9):
        super(LRFinder, self).__init__()
        self.npz_filepath = npz_filepath
        self.run_id = run_id
        self.png_filepath = png_filepath
        self.start_lr, self.end_lr = start_lr, end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []
        self.chosen_lr = 0.0

    def on_train_begin(self, logs=None):
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []
        self.model.saveVariablesNPZ(self.npz_filepath)

    def on_train_batch_begin(self, batch, logs=None):
        if self.step > self.max_steps:
            return
        self.lr = self.exp_annealing(self.step)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def on_train_batch_end(self, batch, logs=None):

        logs = logs or {}
        loss = logs.get('loss')
        step = self.step
        if loss:
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)
            self.lrs.append(self.lr)

            if step == 0 or loss < self.best_loss:
                self.best_loss = loss

            if tf.math.is_nan(smooth_loss):
                print("LOSS BROKE! loss", loss, "smooth_loss", smooth_loss, "self.avg_loss", self.avg_loss, "self.smoothing", self.smoothing)
                step = self.max_steps
                #self.model.stop_training = True

        if step == self.max_steps:
            best_lr = self.plot()
            tf.keras.backend.set_value(self.model.optimizer.lr, best_lr)
            self.chosen_lr = best_lr
            self.model.loadVariablesNPZ(self.npz_filepath)
            os.remove(self.npz_filepath)
            self.model.stop_training = True

        self.step += 1

    def exp_annealing(self, step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.plot(self.lrs, self.losses, color="blue")

        def plot_loss_change(sched, sma=1, n_skip=110, y_lim=(-0.05, 0.05)):
            """
            Plots rate of change of the loss function.
            Parameters:
                sched - learning rate scheduler, an instance of LR_Finder class.
                sma - number of batches for simple moving average to smooth out the curve.
                n_skip - number of batches to skip on the left.
                y_lim - limits for the y axis.
            """
            def take_deri(target):
                derivatives = [0] * (sma + 1)
                for i in range(1 + sma, len(target)):
                    derivative = (target[i] - target[i - sma]) / sma
                    derivatives.append(derivative)
                return derivatives

            derivatives = take_deri(sched.losses)
            derivatives_2 = take_deri(derivatives)
            # derivatives = [0] * (sma + 1)
            # for i in range(1 + sma, len(sched.lrs)):
            #     derivative = (sched.losses[i] - sched.losses[i - sma]) / sma
            #     derivatives.append(derivative)

            lrs, deri, deri2 = sched.lrs[n_skip:], derivatives[n_skip:], derivatives_2[n_skip:]
            best_index = np.argmin(deri)
            best_index2 = np.argmin(deri2)

            ax2 = plt.twinx()
            color = 'grey'
            ax2.set_ylabel("d/loss")  # we already handled the x-label with ax1
            ax2.plot(lrs, deri, color="orange")
            # ax2.plot(lrs, deri2, color="red")
            ax2.scatter([lrs[best_index]], [deri[best_index]], color="orange")
            # ax2.scatter([lrs[best_index2]], [deri2[best_index2]], color="red")
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(y_lim)
            return lrs[best_index]

        best_lr = plot_loss_change(self, sma=20)
        # plt.show()
        plt.savefig(self.png_filepath)
        # if self.run_id == 0:
        #     best_lr = best_lr * 0.1
        return best_lr #* 0.98

def find_lr(model, train_dataset, test_dataset, model_base_path, npz_index, with_short_train, args):
    lr_start = 1e-6
    lr_stop = 0.004
    if args["opt"] == "adam":
        lr_start = 1e-7
        lr_stop = 0.0004
    max_steps = 4000
    if args["model-name"] == "LeNetLike":
        max_steps = 500
    if args["mobnet_alpha"] != None:
        print()
        print("REDUCING MAX LR! Reason: MobileNet")
        print()
        lr_stop /= 10

    print("SEARCHING FOR LR! (FOR", max_steps,"STEPS)")
    lr_schedule = LRFinder(npz_index, model_base_path.format(f"model_q_test{npz_index}.npz"), model_base_path.format(f"model_q_test{npz_index}.png"), start_lr=lr_start, end_lr=lr_stop, max_steps=max_steps)
    model.fit(train_dataset, epochs=50, validation_data=test_dataset, use_multiprocessing=True, verbose=2, validation_freq=1, callbacks=[lr_schedule])
    print()
    print("LR FOUND! LR IS SET TO", lr_schedule.chosen_lr)
    print()

    # if with_short_train:
    #     print("Training shortly")
    #     model.optimizer.lr.assign(lr_schedule.chosen_lr)
    #     model.fit(train_dataset.take(100), epochs=1, validation_data=test_dataset, use_multiprocessing=True, verbose=2, validation_freq=1)
    #
    #     print("SEARCHING FOR LR AGAIN!")
    #     lr_schedule = LRFinder(npz_index+1, model_base_path.format(f"model_q_test{npz_index+1}.npz"), model_base_path.format(f"model_q_test{npz_index+1}.png"), start_lr=lr_start, end_lr=lr_stop, max_steps=max_steps)
    #     model.fit(train_dataset, epochs=50, validation_data=test_dataset, use_multiprocessing=True, callbacks=[lr_schedule])
    #     print()
    #     print("LR FOUND AGAIN! LR IS SET TO", lr_schedule.chosen_lr)
    #     print()
    return lr_schedule.chosen_lr
