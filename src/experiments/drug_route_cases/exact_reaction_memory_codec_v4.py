"""Training geometry validation: retain explicitly supplied known aromatic N-H."""
from run_last_epoch_diagnostic import ROOT
source=(ROOT/'exact_reaction_memory_codec_v3.py').read_text(encoding='utf-8')
old="""        if atom.GetAtomicNum()==7 and atom.GetIsAromatic():
            atom.SetNumExplicitHs(max(atom.GetNumExplicitHs(),src.GetImplicitHCount(),3-src.GetExplicitValence()))"""
new="""        if atom.GetAtomicNum()==7 and atom.GetIsAromatic():
            if i<n:
                atom.SetNumExplicitHs(desc['atoms'][i]['explicit_h'])
            else:
                atom.SetNumExplicitHs(max(atom.GetNumExplicitHs(),src.GetImplicitHCount(),3-src.GetExplicitValence()))"""
assert source.count(old)==1;source=source.replace(old,new)
exec(compile(source,__file__,'exec'),globals())
