"""Train-fitted chemical views attached to the immutable V10 graph cache.

No experimental labels enter RDKit features. Molecular standardization and
split identities remain owned by the accepted input cache.
"""
from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator

import v10_feature_cache as graph
from v10_functional_transfer import require

NAMES = ('MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
         'NumRotatableBonds', 'RingCount', 'NumAromaticRings', 'NumAliphaticRings',
         'NumSaturatedRings', 'NumHeterocycles', 'FractionCSP3', 'HeavyAtomCount',
         'NHOHCount', 'NOCount', 'NumValenceElectrons', 'BertzCT', 'BalabanJ',
         'Chi0v', 'Chi1v', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3')
FEATURE_SPEC = dict(schema='v11_chemical_views_v1', rdkit='2025.09.6',
                    fingerprint=dict(radius=2, fpSize=2048, includeChirality=True),
                    descriptors=list(NAMES), bitorder='little',
                    statistics='all_legal_train_observations_with_multiplicity',
                    standardization='accepted_canonical_SMILES_no_new_standardization')


def tensor_sha(x):
    a = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def chemistry(canonicals):
    require(rdBase.rdkitVersion == FEATURE_SPEC['rdkit'], 'frozen RDKit version')
    require(all(hasattr(Descriptors, n) for n in NAMES), 'descriptor functions unavailable')
    generator = rdFingerprintGenerator.GetMorganGenerator(**FEATURE_SPEC['fingerprint'])
    fingerprints, descriptors, missing = [], [], []
    for text in canonicals:
        require(isinstance(text, str) and text, 'canonical SMILES type')
        mol = Chem.MolFromSmiles(text)
        require(mol is not None and mol.GetNumAtoms() > 0, 'invalid accepted SMILES: '+text)
        bits = np.asarray(generator.GetFingerprintAsNumPy(mol), dtype=np.uint8)
        fingerprints.append(np.packbits(bits, bitorder='little'))
        row, absent = [], []
        for name in NAMES:
            try:
                value = float(getattr(Descriptors, name)(mol))
            except (ValueError, ZeroDivisionError, OverflowError):
                value = math.nan
            flag = not math.isfinite(value)
            row.append(0. if flag else value); absent.append(flag)
        descriptors.append(row); missing.append(absent)
    return (torch.tensor(np.asarray(fingerprints), dtype=torch.uint8),
            torch.tensor(descriptors, dtype=torch.float64),
            torch.tensor(missing, dtype=torch.bool))


def descriptor_statistics(raw, missing, indices):
    x = raw[indices].numpy().copy(); m = missing[indices].numpy()
    medians = np.array([np.median(x[~m[:,j],j]) if (~m[:,j]).any() else 0.
                        for j in range(len(NAMES))])
    x = np.where(m, medians[None, :], x)
    return dict(median=torch.from_numpy(medians), mean=torch.from_numpy(x.mean(0)),
                scale=torch.from_numpy(x.std(0).clip(1e-6)),
                missing_train_count=torch.from_numpy(m.sum(0)),
                all_missing=torch.from_numpy(m.all(0)))


def build(value):
    graph.Data(value)
    rows = value['rows']; canonical = sorted({r['canonical'] for r in rows})
    lookup = {c:i for i,c in enumerate(canonical)}
    indices = torch.tensor([lookup[r['canonical']] for r in rows], dtype=torch.long)
    train = [i for i,r in enumerate(rows) if r['split']=='train']
    fp, desc, missing = chemistry(canonical)
    chemical = dict(spec=FEATURE_SPEC, canonical=canonical, row_indices=indices,
                    fp_packed=fp, descriptors=desc, missing=missing,
                    statistics=descriptor_statistics(desc,missing,indices[train]),
                    rows_sha256=graph.digest(rows),
                    statistics_population_sha256=graph.digest([rows[i] for i in train]))
    # Graph features are referenced separately; never overwrite the accepted cache.
    check(value,chemical)
    data = Data(value,chemical)
    center, factors = robust_fit(data,train)
    chemical['robust_center'] = center
    chemical['robust_factors'] = factors
    return chemical


def check(value, chemical, *, raw_replay=False, robust=False):
    require(chemical['spec']==FEATURE_SPEC and rdBase.rdkitVersion==FEATURE_SPEC['rdkit'], 'chemical feature specification/version')
    rows = value['rows']; canonical=sorted({r['canonical'] for r in rows})
    require(chemical['canonical']==canonical and chemical['rows_sha256']==graph.digest(rows), 'chemical row/canonical identity')
    lookup={c:i for i,c in enumerate(canonical)}
    expected=torch.tensor([lookup[r['canonical']] for r in rows],dtype=torch.long)
    ids=chemical['row_indices']; n=len(canonical)
    require(ids.dtype==torch.long and torch.equal(ids,expected), 'chemical row indices')
    specs={'fp_packed':(torch.uint8,(n,256)), 'descriptors':(torch.float64,(n,24)), 'missing':(torch.bool,(n,24))}
    for key,(dtype,shape) in specs.items():
        x=chemical[key]
        require(x.dtype==dtype and tuple(x.shape)==shape and bool(torch.isfinite(x).all()), 'chemical tensor: '+key)
    require(bool((chemical['descriptors'][chemical['missing']]==0).all()), 'missing descriptor placeholder')
    train=[i for i,r in enumerate(rows) if r['split']=='train']
    require(chemical['statistics_population_sha256']==graph.digest([rows[i] for i in train]), 'chemical train statistics population')
    stats=descriptor_statistics(chemical['descriptors'],chemical['missing'],ids[train])
    require(set(stats)==set(chemical['statistics']) and all(torch.equal(v,chemical['statistics'][k]) for k,v in stats.items()), 'chemical train statistics')
    if 'robust_center' in chemical or 'robust_factors' in chemical:
        width=value['h'].shape[1]+2*len(value['sources'])+2048+48+len(next(iter(value['metadata']['vectors'].values())))
        for key in ('robust_center','robust_factors'):
            x=chemical[key]
            require(x.dtype==torch.float32 and tuple(x.shape)==(width,) and bool(torch.isfinite(x).all()), 'robust statistics shape/finite')
        require(bool((chemical['robust_factors']>=0).all()), 'robust scale sign')
    if raw_replay:
        fresh=chemistry(canonical)
        require(all(torch.equal(x,chemical[k]) for x,k in zip(fresh,('fp_packed','descriptors','missing'))), 'raw SMILES feature replay')
    if robust:
        center,factors=robust_fit(Data(value,chemical),train)
        require(torch.equal(center,chemical['robust_center']) and torch.equal(factors,chemical['robust_factors']), 'RealMLP train statistics')


def save(value, folder, graph_manifest_sha, commit):
    folder.mkdir(parents=True,exist_ok=False)
    chemical=build(value)
    with (folder/'chemical.pt').open('xb') as f: torch.save(chemical,f)
    graph.write(folder/'manifest.json',dict(commit=commit,graph_manifest_sha256=graph_manifest_sha,
                chemical_sha256=graph.sha(folder/'chemical.pt'),spec=FEATURE_SPEC,
                rows_sha256=chemical['rows_sha256'],unique_canonical=len(chemical['canonical']),
                statistics_population_sha256=chemical['statistics_population_sha256']))
    return chemical


def load(value,folder,graph_manifest_sha,commit):
    m=graph.read(folder/'manifest.json')
    require(m['commit']==commit and m['graph_manifest_sha256']==graph_manifest_sha and m['spec']==FEATURE_SPEC
            and m['chemical_sha256']==graph.sha(folder/'chemical.pt'), 'chemical manifest/SHA/source')
    chemical=torch.load(folder/'chemical.pt',map_location='cpu',weights_only=True)
    check(value,chemical)
    require(m['rows_sha256']==chemical['rows_sha256'] and m['unique_canonical']==len(chemical['canonical'])
            and m['statistics_population_sha256']==chemical['statistics_population_sha256'], 'chemical manifest population')
    return chemical


@dataclass
class Episode:
    base: object
    context_fp: torch.Tensor
    context_descriptors: torch.Tensor
    query_fp: torch.Tensor
    query_descriptors: torch.Tensor


class Data(graph.Data):
    def __init__(self,value,chemical,device='cpu'):
        super().__init__(value,device)
        self.chemical=chemical
        packed=chemical['fp_packed'].numpy()
        self.fp=torch.from_numpy(np.unpackbits(packed,axis=1,bitorder='little')).to(device)
        stats=chemical['statistics']; missing=chemical['missing']
        desc=torch.where(missing,stats['median'][None,:],chemical['descriptors'])
        self.descriptors=torch.cat(((desc-stats['mean'])/stats['scale'],missing.double()),-1).float().to(device)
        self.chemical_indices=chemical['row_indices'].to(device)
        self.context_index={t:{self.rows[i]['sample_id']:i for i in self.by[t,'train']} for t in self.tasks}

    def episode(self,task,query,**kwargs):
        base=super().episode(task,query,**kwargs)
        by_id=self.context_index[task]
        context=[by_id[r.sample_id] for r in base.context_rows]
        ci,qi=self.chemical_indices[context],self.chemical_indices[query]
        return Episode(base,self.fp[ci].float(),self.descriptors[ci],self.fp[qi].float(),self.descriptors[qi])

    def flat_rows(self,indices):
        n=len(indices); mask=self.h.new_ones(n,len(self.sources))
        for j,i in enumerate(indices):
            if self.rows[i]['task'] in self.sources: mask[j,self.sources.index(self.rows[i]['task'])]=0
        metadata=self.h.new_tensor([self.value['metadata']['vectors'][self.rows[i]['task']] for i in indices])
        ci=self.chemical_indices[indices]
        return torch.cat((self.h[indices],self.p[indices]*mask,mask,self.fp[ci].float(),self.descriptors[ci],metadata),-1)


def robust_fit(data,train):
    """RealMLP-TD-S median/IQR with its documented range fallback; train only."""
    # Bounded chunks avoid a full GPU copy and quantile's large temporary tensors.
    arrays=[data.flat_rows(train[i:i+2048]).detach().cpu().numpy() for i in range(0,len(train),2048)]
    x=np.concatenate(arrays)
    q=np.quantile(x,[.25,.5,.75],axis=0)
    scale=q[2]-q[0]; fallback=.5*(x.max(0)-x.min(0))
    scale=np.where(scale==0,fallback,scale)
    factors=np.zeros_like(scale); np.divide(1.,scale+1e-30,out=factors,where=scale!=0)
    return torch.tensor(q[1],dtype=torch.float32),torch.tensor(factors,dtype=torch.float32)
