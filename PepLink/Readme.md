# PepLink ✨

PepLink ✨ can do transformation between **amino acid sequences** and **SMILES/SELFIES strings**, which supports:  
- 420 non-canonical amino acids 
- 11 kinds of intra-molecular bonds
- 

**[all_aa_smiles_new_handcrafted.csv](./all_aa_smiles_new_handcrafted.csv)** contains the SMILES strings of 460 amino acids

## Example Usage
### Amino Acid Sequence → SMILES/SELFIES
Example data structure (same as from [DBAASP](https://dbaasp.org/home))  
* note: a lot irrelevant information is omitted 
```json
{
    "id": 57,
    "dbaaspId": "DBAASPR_57",
    "sequence": "VTCDILSVEAKGVKLNDAACAAHCLFRGRSGGYCNGKRVCVCR",
    "sequenceLength": 43,
    "nTerminus": null,
    "cTerminus": null,
    "intrachainBonds": [
        {
            "id": 12,
            "position1": 3,
            "position2": 34,
            "type": {
                "id": 58,
                "name": "DSB",
                "description": "Disulfide Bond"
            },
            "cycleType": {
                "id": 4,
                "intraChainBondTypeId": 0,
                "name": "CST",
                "description": "Cystine"
            },
            "chainParticipating": {
                "id": 2,
                "name": "SSB",
                "description": "Sidechain-Sidechain Bond"
            },
            "note": null
        },
        {
            "id": 13,
            "position1": 20,
            "position2": 40,
            "type": {
                "id": 58,
                "name": "DSB",
                "description": "Disulfide Bond"
            },
            "cycleType": {
                "id": 4,
                "intraChainBondTypeId": 0,
                "name": "CST",
                "description": "Cystine"
            },
            "chainParticipating": {
                "id": 2,
                "name": "SSB",
                "description": "Sidechain-Sidechain Bond"
            },
            "note": null
        },
        {
            "id": 14,
            "position1": 24,
            "position2": 42,
            "type": {
                "id": 58,
                "name": "DSB",
                "description": "Disulfide Bond"
            },
            "cycleType": {
                "id": 4,
                "intraChainBondTypeId": 0,
                "name": "CST",
                "description": "Cystine"
            },
            "chainParticipating": {
                "id": 2,
                "name": "SSB",
                "description": "Sidechain-Sidechain Bond"
            },
            "note": null
        }
    ],
    "interchainBonds": [],
    "coordinationBonds": [],
    "unusualAminoAcids": [],
    "targetActivities": [
        {
            "id": 132,
            "targetSpecies": {
                "name": "Staphylococcus aureus ATCC 6538"
            },
            "activityMeasureGroup": {
                "name": "MIC"
            },
            "activityMeasureValue": "MIC",
            "concentration": "6.25",
            "unit": {
                "name": "µg/ml",
                "description": ""
            },
            "ph": "",
            "ionicStrength": "",
            "saltType": "",
            "medium": {
                "name": "AM3",
                "description": "Antibiotic medium 3"
            },
            "cfu": "5E3",
            "cfuGroup": {
                "name": "1E3 - 1E4"
            },
            "note": "",
            "reference": "1",
            "activity": 6.25
        },
        ... omitted ...
        {
            "id": 61573,
            "targetSpecies": {
                "name": "Streptococcus pyogenes 308A",
                "description": ""
            },
            "activityMeasureGroup": {
                "name": "MIC"
            },
            "activityMeasureValue": "MIC",
            "concentration": "1.6",
            "unit": {
                "name": "µg/ml",
                "description": ""
            },
            "ph": "",
            "ionicStrength": "",
            "saltType": "",
            "medium": {
                "name": "AM3A",
                "description": "Antibiotic Medium 3 Agar"
            },
            "cfu": "",
            "cfuGroup": null,
            "note": "",
            "reference": "2",
            "activity": 1.6
        }
    ],
    "smiles": null,
    "smilesImageUrl": null
}
```
```python
from PepLink import *

# load SMILES of amino acids
aa_smiles_dict = get_aa_smiles_dict('./Data/all_aa_smiles_new_handcrafted.csv')
# TODO

```
