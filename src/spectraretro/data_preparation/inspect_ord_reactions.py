from __future__ import annotations

import gzip
import sys
from pathlib import Path

from ord_schema.proto import dataset_pb2
from ord_schema.proto import reaction_pb2


def identifier_value(identifiers, enum_name: str) -> str:
    enum = reaction_pb2.ReactionIdentifier.IdentifierType.Value(enum_name)
    for ident in identifiers:
        if ident.type == enum:
            return ident.value
    return ""


def main() -> None:
    path = Path(sys.argv[1])
    dataset = dataset_pb2.Dataset()
    with gzip.open(path, "rb") as f:
        dataset.ParseFromString(f.read())
    print(dataset.dataset_id, dataset.name, len(dataset.reactions))
    print(dataset.description[:500].replace("\n", " "))
    for i, reaction in enumerate(dataset.reactions[:10], start=1):
        rxn = identifier_value(reaction.identifiers, "REACTION_SMILES")
        cx = identifier_value(reaction.identifiers, "REACTION_CXSMILES")
        name = identifier_value(reaction.identifiers, "NAME")
        print(f"{i}\t{name}\t{rxn or cx}")


if __name__ == "__main__":
    main()
