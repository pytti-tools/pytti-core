import math
import pandas as pd
from pytti.Notebook import tqdm
from pytti import format_input
from scipy.signal import savgol_filter
from torch import optim, nn

from pytti.Image import DifferentiableImage

import einops as eo
from loguru import logger

# deprecate this
from labellines import labelLines


def unpack_dict(D, n=2):
    """
    Given a dictionary D and a number n, return a tuple of n dictionaries,
    each containing the same keys as D and values corresponding to those
    values of D at the corresponding index

    :param D: a dictionary
    :param n: number of samples to draw, defaults to 2 (optional)
    :return: A tuple of dictionaries.
    """
    ds = [{k: V[i] for k, V in D.items()} for i in range(n)]
    return tuple(ds)


# this only gets used in the plot_losses method below.
# deprecate (tensorboard)
def smooth_dataframe(df, window_size):
    """applies a moving average filter to the columns of df"""
    smoothed_df = pd.DataFrame().reindex_like(df)
    for key in df.columns:
        smoothed_df[key] = savgol_filter(df[key], window_size, 2, mode="nearest")
    return smoothed_df


class DirectImageGuide:
    """
    Image guide that uses an optimizer and torch autograd to optimize an image representation
    Based on the BigGan+CLIP algorithm by advadnoun (https://twitter.com/advadnoun)
    image_rep: (DifferentiableImage) image representation
    embedder: (Module)               image embedder
    optimizer: (Class)               optimizer class to use. Defaults to Adam
    all other arguments are passed as kwargs to the optimizer.
    """

    def __init__(
        self,
        image_rep: DifferentiableImage,
        embedder: nn.Module,
        optimizer: optim.Optimizer = None,
        lr: float = None,
        **optimizer_params,
    ):
        self.image_rep = image_rep
        self.embedder = embedder
        if lr is None:
            lr = image_rep.lr
        optimizer_params["lr"] = lr
        self.optimizer_params = optimizer_params
        if optimizer is None:
            self.optimizer = optim.Adam(image_rep.parameters(), **optimizer_params)
        else:
            self.optimizer = optimizer
        self.dataframe = []

    def run_steps(
        self,
        n_steps,
        prompts,
        interp_prompts,
        loss_augs,
        stop=-math.inf,
        interp_steps=0,
        i_offset=0,
        skipped_steps=0,
    ):
        """
        runs the optimizer
        prompts: (ClipPrompt list) list of prompts
        n_steps: (positive integer) steps to run
        returns: the number of steps run
        """
        for i in tqdm(range(n_steps)):
            self.update(i + i_offset, i + skipped_steps)
            losses = self.train(
                i + skipped_steps,
                prompts,
                interp_prompts,
                loss_augs,
                interp_steps=interp_steps,
            )
            if losses["TOTAL"] <= stop:
                break
        return i + 1

    def set_optim(self, opt=None):
        if opt is not None:
            self.optimizer = opt
        else:
            self.optimizer = optim.Adam(
                self.image_rep.parameters(), **self.optimizer_params
            )

    def clear_dataframe(self):
        self.dataframe = []

    # deprecate (tensorboard)
    def plot_losses(self, axs):
        def plot_dataframe(df, ax, legend=False):
            keys = list(df)
            keys.sort(reverse=True, key=lambda k: df[k].iloc[-1])
            ax.clear()
            df[keys].plot(ax=ax, legend=legend)
            if legend:
                ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            ax.tick_params(
                labelbottom=True,
                labeltop=False,
                labelleft=True,
                labelright=False,
                bottom=True,
                top=False,
                left=True,
                right=False,
            )
            last_x = df.last_valid_index()
            lines = ax.get_lines()

            colors = [l.get_color() for l in lines]
            labels = [l.get_label() for l in lines]
            ax.relim()
            ax.autoscale_view()

            labelLines(ax.get_lines(), align=False)
            return dict(zip(labels, colors))

        dfs = self.dataframe[:]
        if dfs != []:
            dfs[0] = smooth_dataframe(dfs[0], 17)
        for i, (df, ax) in enumerate(zip(dfs, axs)):
            if len(df.index) < 2:
                return False
            # m = df.apply(lambda col: col.first_valid_index())
            # print(m)
            # print(df.lookup(m, m.index))
            # rel_loss = (df-df.lookup(m, m.index))
            if not df.empty:
                plot_dataframe(df, ax, legend=i == 0)
            ax.set_ylabel("Loss")
            ax.set_xlabel("Step")
        return True

    def update(self, i, stage_i):
        """
        update hook called ever step
        """
        pass

    def train(
        self,
        i,
        prompts,
        interp_prompts,
        loss_augs,
        interp_steps=0,
        save_loss=True,
        gradient_accumulation_steps: int = 4,
    ):
        """
        steps the optimizer
        promts: (ClipPrompt list) list of prompts
        """
        self.optimizer.zero_grad()
        z = self.image_rep.decode_training_tensor()
        logger.debug(z.shape)  # [1, 3, height, width]
        losses = []

        aug_losses = {
            aug: aug(format_input(z, self.image_rep, aug), self.image_rep)
            for aug in loss_augs
        }

        image_augs = self.image_rep.image_loss()
        image_losses = {aug: aug(self.image_rep) for aug in image_augs}

        if self.embedder is not None:
            image_embeds, offsets, sizes = self.embedder(self.image_rep, input=z)
            logger.debug(
                image_embeds.shape
            )  # [1, 40, latent_dim] # vqgan latent_dim=512
            logger.debug(offsets.shape)  # [1, 40, 2]
            logger.debug(sizes.shape)  # [1, 40, 2]

        # reshape for gradient accumulation minibatches
        image_embeds_batched = eo.rearrange(
            image_embeds,
            "batch (cuts mb) latent -> (batch mb) cuts latent",
            mb=gradient_accumulation_steps,
        )
        offsets_batched = eo.rearrange(
            offsets,
            "batch (cuts mb) xy -> (batch mb) cuts xy",
            mb=gradient_accumulation_steps,
        )
        sizes_batched = eo.rearrange(
            sizes,
            "batch (cuts mb) xy -> (batch mb) cuts xy",
            mb=gradient_accumulation_steps,
        )

        total_loss = 0
        for mb_i in range(gradient_accumulation_steps):
            logger.debug(mb_i)
            image_embeds = image_embeds_batched[mb_i, ...].unsqueeze(0)
            offsets = offsets_batched[mb_i, ...].unsqueeze(0)
            sizes = sizes_batched[mb_i, ...].unsqueeze(0)
            logger.debug(
                image_embeds.shape
            )  # [1, 40, latent_dim] # vqgan latent_dim=512
            logger.debug(offsets.shape)  # [1, 40, 2]
            logger.debug(sizes.shape)  # [1, 40, 2]

            t = 1
            interp_losses = [0]
            if i < interp_steps:
                t = i / interp_steps
                interp_losses = [
                    prompt(
                        format_input(image_embeds, self.embedder, prompt),
                        format_input(offsets, self.embedder, prompt),
                        format_input(sizes, self.embedder, prompt),
                    )[0]
                    * (1 - t)
                    for prompt in interp_prompts
                ]

            prompt_losses = {
                prompt: prompt(
                    format_input(image_embeds, self.embedder, prompt),
                    format_input(offsets, self.embedder, prompt),
                    format_input(sizes, self.embedder, prompt),
                )
                for prompt in prompts
            }

            losses, losses_raw = zip(
                *map(unpack_dict, [prompt_losses, aug_losses, image_losses])
                # *map(unpack_dict, [prompt_losses])
            )

            losses = list(losses)
            losses_raw = list(losses_raw)
            for v in prompt_losses.values():
                v[0].mul_(t)

            # total_loss = sum(map(lambda x: sum(x.values()), losses)) + sum(interp_losses)
            total_loss_mb = sum(map(lambda x: sum(x.values()), losses)) + sum(
                interp_losses
            )
            logger.debug(f"total_loss (mb_i): {total_loss_mb}")
            total_loss_mb /= gradient_accumulation_steps
            # losses_raw.append({"TOTAL": total_loss}) # before calling backward?
            logger.debug(
                f"total_loss / mb: {total_loss_mb}"
            )  # ... doesn't matter. 0.8727 final output ....8745 now. and now 8709?
            # total_loss.backward()
            total_loss_mb.backward(retain_graph=True)
            total_loss += total_loss_mb
            logger.debug(".backward() called successfully")

        # logger.debug("computing aug losses")
        # losses, losses_raw = zip(
        #        *map(unpack_dict, [aug_losses, image_losses])
        #    )
        # if len(losses) > 0:
        #    logger.debug(len(losses))
        #    logger.debug(losses.shape) # tuples
        #    total_loss += sum(map(lambda x: sum(x.values()), losses))
        #    total_loss.backward() # where does zero_grad() get called?

        losses_raw.append({"TOTAL": total_loss})  # before calling backward?
        logger.debug(f"total_loss: {total_loss}")
        self.optimizer.step()
        self.image_rep.update()
        # if t != 0:
        #  for v in prompt_losses.values():
        #    v[0].div_(t)
        if save_loss:
            if not self.dataframe:
                self.dataframe = [
                    pd.DataFrame({str(k): float(v) for k, v in loss.items()}, index=[i])
                    for loss in losses_raw
                ]
                for df in self.dataframe:
                    df.index.name = "Step"
            else:
                for j, (df, loss) in enumerate(zip(self.dataframe, losses_raw)):
                    self.dataframe[j] = df.append(
                        pd.DataFrame(
                            {str(k): float(v) for k, v in loss.items()}, index=[i]
                        ),
                        ignore_index=False,
                    )
                    self.dataframe[j].name = "Step"

        return {"TOTAL": float(total_loss)}
