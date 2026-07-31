from rdkit import Chem
from rdkit.Chem import Descriptors
def smiles2molecular_weight(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        MW = Descriptors.MolWt(molecule)
    except:
        return None
    return MW

def calculate_mgkg(smiles, res):
    MW = smiles2molecular_weight(smiles)
    if MW is None:
        return None
    mgkg = MW / (10**res * 0.001)
    return mgkg
def count_parameters(model):
    r'''Calculate the number of parameters for a model.

    Args:
        model (torch.nn.Module): A neural network module.
    '''
    trainable_params = 0
    non_trainable_params = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        else:
            non_trainable_params += p.numel()
    print('='*40)
    print('Total Params:', trainable_params + non_trainable_params)
    print('Trainable Params:', trainable_params)
    print('Non-trainable Params:', non_trainable_params)