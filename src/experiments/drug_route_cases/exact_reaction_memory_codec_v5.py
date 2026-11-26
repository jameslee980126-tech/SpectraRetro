"""Reject sanitizer changes to known bonds as an invalid sample, not a queue failure."""
import exact_reaction_memory_codec_v4 as previous
source=previous.source
old="            assert str(mol.GetBondBetweenAtoms(a,b).GetBondType())==kind"
new="            if str(mol.GetBondBetweenAtoms(a,b).GetBondType())!=kind:return None,'sanitize_changed_known_bond_order'"
assert source.count(old)==1;source=source.replace(old,new)
exec(compile(source,__file__,'exec'),globals())
