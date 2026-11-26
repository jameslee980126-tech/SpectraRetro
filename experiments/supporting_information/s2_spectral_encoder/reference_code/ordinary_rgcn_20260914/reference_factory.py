"""Current product-only FSpec versus the original ordinary product RGCN."""
from torchdrug import models
from torchdrug.models.fspec_faithful_rc import FSpecFaithfulReactionCenter
def make_model(arm,dataset,width=512):
    if arm=='full_fspec':
        return FSpecFaithfulReactionCenter(dataset.node_feature_dim,[width]*3,dataset.edge_feature_dim)
    if arm=='ordinary_rgcn':
        # Same encoder configuration as original unknown_class/stage1/train.py.
        # Bond relation types enter propagation; raw bond features enter the original task readout.
        # Do NOT add the custom edge encoder/readout used by prior local_rgcn.
        # dataset.num_bond_type counts observed types, which can omit a type in a small subset.
        # Use the actual graph schema, as the existing task also validates.
        return models.RGCN(input_dim=dataset.node_feature_dim,hidden_dims=[width]*3,
            batch_norm=False,short_cut=True,num_relation=int(dataset[0]['graph'][1].num_relation),concat_hidden=False)
    raise ValueError(arm)
