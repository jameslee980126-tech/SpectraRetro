"""Local Open Babel decoder; no reference molecules are read."""
import json,re,sys,os,ctypes
from pathlib import Path
# Windows format plugins need their runtime DLL directory on PATH before import.
babel_bin=Path(sys.prefix)/'Library/bin'
os.environ['PATH']=str(babel_bin)+os.pathsep+os.environ['PATH']
os.environ['BABEL_LIBDIR']=str(babel_bin)
os.environ['BABEL_DATADIR']=str(Path(sys.prefix)/'share/openbabel')
_dll_dir=os.add_dll_directory(str(babel_bin))
_formats=ctypes.WinDLL(str(babel_bin/'formats_common.obf'))
import numpy as np
from rdkit import Chem, RDLogger
from openbabel import openbabel as ob
RDLogger.DisableLog('rdApp.*')
ob.obErrorLog.SetOutputLevel(-1)
ELEMENTS='C N O S B Br Cl P I F Mg Si Sn Zn Cu Se H Li Na Cr Mn Ge Ru Ag Te Pt'.split()

def decode(x,t):
    text=str(len(t))+'\n\n'+''.join(f'{ELEMENTS[int(i)]} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f}\n' for p,i in zip(x,t))
    conv=ob.OBConversion()
    if not conv.SetInAndOutFormats('xyz','smi'):raise RuntimeError('Open Babel XYZ/SMILES plugin unavailable')
    mol=ob.OBMol()
    if not conv.ReadString(mol,text):return dict(raw=None,smiles=None,reason='xyz_decode_failed')
    raw=conv.WriteString(mol).split('\t')[0].strip()
    if not raw:raise RuntimeError('Open Babel returned empty SMILES after successful XYZ read')
    restored=re.sub(r'\[(C|N|O|c|n|o|S|s|P)\]',r'\1',raw)
    m=Chem.MolFromSmiles(restored)
    reason=None
    if m is None:reason='sanitize_failed'
    elif len(Chem.GetMolFrags(m))!=1:reason='disconnected_fragment'
    elif any(a.GetNumRadicalElectrons() for a in m.GetAtoms()):reason='residual_radicals'
    elif any(a.GetFormalCharge() for a in m.GetAtoms()):reason='charged_candidate_unsupported'
    return dict(raw=raw,smiles=Chem.MolToSmiles(m,isomericSmiles=True) if reason is None else None,reason=reason)

def main():
    source,out=map(Path,sys.argv[1:3]);npz=np.load(source)
    rows=[];offset=0
    for i,n in enumerate(npz['length']):
        rows.append(dict(index=i,**decode(npz['xyz'][offset:offset+n],npz['element'][offset:offset+n])))
        offset+=n
    out.write_text(json.dumps(dict(openbabel=ob.OBReleaseVersion(),rows=rows),indent=2),encoding='utf-8')

if __name__=='__main__':main()
