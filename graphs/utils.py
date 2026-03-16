import rdkit
import rdkit.Chem as Chem
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as pyg_DataLoader
from torch_geometric.nn import GCNConv, global_add_pool, BatchNorm

structural_smarts = {
    # chirality
    "Specified chiral carbon": "[$([#6X4@](*)(*)(*)*),$([#6X4@H](*)(*)*)]",  # Matches carbons whose chirality is specified (clockwise or anticlockwise) Will not match molecules whose chirality is unspecified b ut that could otherwise be considered chiral. Also,therefore won't match molecules that would be chiral due to an implicit connection (i.e.i mplicit H).
    # connectivity
    "Quaternary Nitrogen": "[$([NX4+]),$([NX4]=*)]",  # Hits non-aromatic Ns.
    "S double-bonded to Carbon": "[$([SX1]=[#6])]",  # Hits terminal (1-connected S)
    "Triply bonded N": "[$([NX1]#*)]",
    "Divalent Oxygen": "[$([OX2])]",
    # chains and branching
    "Long_chain groups": "[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]",  # Aliphatic chains at-least 8 members long.
    "Carbon_isolating": "[$([#6+0]);!$(C(F)(F)F);!$(c(:[!c]):[!c])!$([#6]=,#[!#6])]",  # This definition is based on that in CLOGP, so it is a charge-neutral carbon, which is not a CF3 or an aromatic C between two aromati c hetero atoms eg in tetrazole, it is not multiply bonded to a hetero atom.
    # rotation
    "Rotatable bond": "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]",  # An atom which is not triply bonded and not one-connected i.e.terminal connected by a single non-ring bond to and equivalent atom. Note that logical operators can be applied to bonds ("-&!@"). Here, the overall SMARTS consists of two atoms and one bond. The bond is "site and not ring". *#* any atom triple bonded to any atom. By enclosing this SMARTS in parentheses and preceding with $, this enables us to use $(*#*) to write a recursive SMARTS using that string as an atom primitive. The purpose is to avoid bonds such as c1ccccc1-C#C which wo be considered rotatable without this specification.
    # cyclic features
    "Bicyclic": "[$([*R2]([*R])([*R])([*R]))].[$([*R2]([*R])([*R])([*R]))]",  # Bicyclic compounds have 2 bridgehead atoms with 3 arms connecting the bridgehead atoms.
    "Ortho": "*-!:aa-!:*",  # Ortho-substituted ring
    "Meta": "*-!:aaa-!:*",  # Meta-substituted ring
    "Para": "*-!:aaaa-!:*",  # Para-substituted ring
    "Acylic-bonds": "*!@*",
    "Single bond and not in a ring": "*-!@*",
    "Non-ring atom": "[!R]",
    "Ring atom": "[R]",
    "Macrocycle groups": "[r;!r3;!r4;!r5;!r6;!r7]",
    "S in aromatic 5-ring with lone pair": "[sX2r5]",
    "Aromatic 5-Ring O with Lone Pair": "[oX2r5]",
    # "N in 5-sided aromatic ring": "[nX2r5]",
    "Spiro-ring center": "[X4;R2;r4,r5,r6](@[r4,r5,r6])(@[r4,r5,r6])(@[r4,r5,r6])@[r4,r5,r6]",  # rings size 4-6
    "N in 5-ring arom": "[$([nX2r5]:[a-]),$([nX2r5]:[a]:[a-])]",  # anion
    "CIS or TRANS double bond in a ring": "*/,\[R]=;@[R]/,\*",  # An isomeric SMARTS consisting of four atoms and three bonds.
    "CIS or TRANS double or aromatic bond in a ring": "*/,\[R]=,:;@[R]/,\*",
    "Unfused benzene ring": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",  # To find a benzene ring which is not fused, we write a SMARTS of 6 aromatic carbons in a ring where each atom is only in one ring:
    "Multiple non-fused benzene rings": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1.[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
    "Fused benzene rings": "c12ccccc1cccc2",
}

functional_group_smarts = {
    # carbonyl
    "Carbonyl group": "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]",  # Hits either resonance structure
    "Aldehyde": "[CX3H1](=O)[#6]",  # -al
    "Amide": "[NX3][CX3](=[OX1])[#6]",  # -amide
    "Carbamate": "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",  # Hits carbamic esters, acids, and zwitterions
    "Carboxylate Ion": "[CX3](=O)[O-]",  # Hits conjugate bases of carboxylic, carbamic, and carbonic acids.
    "Carbonic Acid or Carbonic Ester": "[CX3](=[OX1])(O)O",  # Carbonic Acid, Carbonic Ester, or combination
    "Carboxylic acid": "[CX3](=O)[OX1H0-,OX2H1]",
    "Ester Also hits anhydrides": "[#6][CX3](=O)[OX2H0][#6]",  # won't hit formic anhydride.
    "Ketone": "[#6][CX3](=O)[#6]",  # -one
    # ether
    "Ether": "[OD2]([#6])[#6]",
    # hydrogen atoms
    "Mono-Hydrogenated Cation": "[+H]",  # Hits atoms that have a positive charge and exactly one attached hydrogen:  F[C+](F)[H]
    "Not Mono-Hydrogenated": "[!H1]",  # Hits atoms that don't have exactly one attached hydrogen.
    # amide
    "Amidinium": "[NX3][CX3]=[NX3+]",
    "Cyanamide": "[NX3][CX2]#[NX1]",
    # amine
    "Primary or secondary amine, not amide": "[NX3;H2,H1;!$(NC=O)]",  # Not ammonium ion (N must be 3-connected), not ammonia (H count can't be 3). Primary or secondary is specified by N's H-count (H2 &amp; H1 respectively).  Also note that "&amp;" (and) is the dafault opperator and is higher precedence that "," (or), which is higher precedence than ";" (and). Will hit cyanamides and thioamides
    "Enamine": "[NX3][CX3]=[CX3]",
    "Enamine or Aniline Nitrogen": "[NX3][$(C=C),$(cc)]",
    # azo
    "Azole": "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",  # 5 member aromatic heterocycle w/ 2double bonds. contains N &amp; another non C (N,O,S)  subclasses are furo-, thio-, pyrro-  (replace
    # hydrazine
    "Hydrazine H2NNH2": "[NX3][NX3]",
    # hydrazone
    "Hydrazone C=NNH2": "[NX3][NX2]=[*]",
    # imine
    "Substituted imine": "[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]",  # Schiff base
    "Substituted or un-substituted imine": "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
    "Iminium": "[NX3+]=[CX3]",
    # imide
    "Unsubstituted dicarboximide": "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
    "Substituted dicarboximide": "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])",
    # nitrate
    "Nitrate group": "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",  # Also hits nitrate anion
    # nitrile
    "Nitrile": "[NX1]#[CX2]",
    # nitro
    "Nitro group": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",   #Hits both forms.
    # hydroxyl (includes alcohol, phenol)
    "Hydroxyl": "[OX2H]",
    "Hydroxyl in Alcohol": "[#6][OX2H]",
    "Enol": "[OX2H][#6X3]=[#6]",
    "Phenol": "[OX2H][cX3]:[c]",
    # thio groups (thio-, thi-, sulpho-, marcapto-)
    "Carbo-Thioester": "S([#6])[CX3](=O)[#6]",
    "Thio analog of carbonyl": "[#6X3](=[SX1])([!N])[!N]",  # Where S replaces O.  Not a thioamide.
    "Thiol, Sulfide or Disulfide Sulfur": "[SX2]",
    "Thioamide": "[NX3][CX3]=[SX1]",
    # sulfide
    "Sulfide": "[#16X2H0]",  # -alkylthio  Won't hit thiols. Hits disulfides.
    "Mono-sulfide": "[#16X2H0][!#16]",  # alkylthio- or alkoxy- Won't hit thiols. Won't hit disulfides.
    "Two Sulfides": "[#16X2H0][!#16].[#16X2H0][!#16]",  # Won't hit thiols. Won't hit mono-sulfides. Won't hit disulfides.
    "Sulfone": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",  # Hits all sulfones, including heteroatom-substituted sulfones:  sulfonic acid, sulfonate, sulfuric acid mono- &amp; di- esters, sulfamic acid, sulfamate, sulfonamide... Hits Both Depiction Forms.
    "Sulfonamide": "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",  # (sulf drugs)  Won't hit sulfamic acid or sulfamate. Hits Both Depiction Forms.
    # sulfoxide
    "Sulfoxide": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",  # ( sulfinyl, thionyl ) Analog of carbonyl where S replaces C. Hits all sulfoxides, including heteroatom-substituted sulfoxides, dialkylsulfoxides carbo-sulfoxides, sulfinate, sulfinic acids... Hits Both Depiction Forms. Won't hit sulfones.
    # halide (-halo -fluoro -chloro -bromo -iodo)
    "Any carbon attached to any halogen": "[#6][F,Cl,Br,I]",
    # Halogen
    "Halogen": "[F,Cl,Br,I]",
    # Three_halides groups
    "Three_halides groups": "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]",  # Hits SMILES that have three halides.
}

def match_patterns(mol, smarts):
    """

    :param mol: RDKIT mol object
    :param smarts: dict: {"name": "SMARTS"}
    :return: torch.Tensor (n_atoms x n_patterns): one hot tensor of functional group membership
    """

    x = torch.zeros(len(smarts), len(mol.GetAtoms()))
    for i, pattern in enumerate(smarts.values()):
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
        # collapse tuple of tuple into unique list of atom indices
        atoms = list(set(sum(matches, ())))
        x[i][atoms] = 1

    return x.T

def atom_props(atom, atomic_property_values):

    x = []

    atom_types = atomic_property_values[0]
    vec = [0] * len(atom_types)
    vec[atom_types.index(atom.GetSymbol())] = 1
    x += vec

    degrees = atomic_property_values[1]
    vec = [0] * len(degrees)
    vec[degrees.index(atom.GetDegree())] = 1
    x += vec

    total_degrees = atomic_property_values[2]
    vec = [0] * len(total_degrees)
    vec[total_degrees.index(atom.GetTotalDegree())] = 1
    x += vec

    explicit_valences = atomic_property_values[3]
    vec = [0] * len(explicit_valences)
    vec[explicit_valences.index(atom.GetValence(Chem.rdchem.ValenceType.EXPLICIT))] = 1
    x += vec

    implicit_valences = atomic_property_values[4]
    vec = [0] * len(implicit_valences)
    vec[implicit_valences.index(atom.GetValence(Chem.rdchem.ValenceType.IMPLICIT))] = 1
    x += vec

    total_valences = atomic_property_values[5]
    vec = [0] * len(total_valences)
    vec[total_valences.index(atom.GetTotalValence())] = 1
    x += vec

    implicit_Hs = atomic_property_values[6]
    vec = [0] * len(implicit_Hs)
    vec[implicit_Hs.index(atom.GetNumImplicitHs())] = 1
    x += vec

    total_Hs = atomic_property_values[7]
    vec = [0] * len(total_Hs)
    vec[total_Hs.index(atom.GetTotalNumHs())] = 1 
    x += vec

    formal_charges = atomic_property_values[8]
    vec = [0] * len(formal_charges)
    vec[formal_charges.index(atom.GetFormalCharge())] = 1
    x += vec

    hybridizations = atomic_property_values[9]
    vec = [0] * len(hybridizations)
    vec[hybridizations.index(atom.GetHybridization().name)] = 1
    x += vec

    return x

def atom_featurizer(mol, atomic_property_values, structural_feats: bool = True, functional_feats: bool = True):

    x = []
    for atom in mol.GetAtoms():
        x_ = atom_props(atom, atomic_property_values)
        x.append(x_)
    x = torch.tensor(x)

    if structural_feats:
        x_struc = match_patterns(mol, structural_smarts)
        x = torch.cat((x, x_struc), dim=1)

    if functional_feats:
        x_func = match_patterns(mol, functional_group_smarts)
        x = torch.cat((x, x_func), dim=1)

    return x

def molecular_graph_featurizer(smiles: str, y=None, atomic_property_values=None, structural_feats: bool = True, functional_feats: bool = True):

    y = torch.tensor([y]).to(torch.float32)

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # RDKIT Atom featurization
    x = atom_featurizer(mol, atomic_property_values, structural_feats, functional_feats)

    # Edge featurization
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_indices += [[i, j], [j, i]]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)

    # Sort indices.
    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]

    if torch.isnan(x).any():
        return smiles
        # raise ValueError(f"Featurizing {smiles} gave nan(s)")

    graph = Data(x=x, edge_index=edge_index, smiles=smiles, y=y)

    return graph