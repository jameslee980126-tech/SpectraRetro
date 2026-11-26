import torch

from tqdm import tqdm
from rdkit import RDLogger

from torch import Tensor
from typing import Sequence

from . import BaseGenerativeModule
from src.data.batch_class import GraphBatch
from src.data.transforms import dense2smiles
from src.utils import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)
RDLogger.DisableLog('rdApp.*')

__all__ = ['RetroGenerativeModule']

class RetroGenerativeModule(BaseGenerativeModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(
            self,
            batch: GraphBatch
        ) -> tuple[Tensor, dict[str, float]]:
        if isinstance(batch, Sequence):
            batch = batch[0]

        t, model_input, target = self.sde(batch)
        pred = self.net(
            t=t, node_mask=batch.node_mask,
            **model_input
        )
        loss, log_out = self.sde.loss(
            t, batch.node_mask, pred, target
        )

        
        if torch.isnan(loss):
            log.info("Loss NAN on step ", self.global_step)
            loss = loss * 0

        return loss, {'loss': loss, 'bsz': t.size(0)} | log_out


    # -------# Evaluating #-------- #
    def sampling(
            self,
            batch: GraphBatch | Sequence,
            is_val: bool
        ) -> None:
        save_store = self.val_store if is_val else self.test_store
        top_k = max(save_store.top_ks)
        if isinstance(batch, Sequence):
            batch, aug_batch = batch
        else:
            aug_batch = [batch] * top_k
        assert len(aug_batch) == top_k
        
        # len(r_smiles_list) == len(p_smiles_list) == batch_size

        r_smiles_list = dense2smiles(
            batch.r_X, batch.r_E, batch.node_mask,
            self.x_dec, self.e_dec, canonical=True
        )
        p_smiles_list = dense2smiles(
            batch.p_X, batch.p_E, batch.node_mask,
            self.x_dec, self.e_dec, canonical=True
        )

        gen_smiles_list = []

        for batch in tqdm(aug_batch, leave=False, dynamic_ncols=True):
            generated = self.sde.sampling(net=self.net, batch=batch, is_val=is_val)
            gen_smiles = dense2smiles(
                generated.r_X, generated.r_E, batch.node_mask,
                self.x_dec, self.e_dec, canonical=True
            )

            gen_smiles_list.append(gen_smiles)

        save_store.store(p_smiles_list, gen_smiles_list, r_smiles_list, None)
                


