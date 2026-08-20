# Chemical structure audit v4

Generated (UTC): 2026-08-20T07:34:52+00:00

## Results
- Input rows: 15,535
- Valid parsed structures: 15,535
- Unique InChIKeys: 10,726
- Duplicate InChIKey groups: 2,448
- Records in duplicate InChIKey groups: 7,257
- Parse-status counts: {'ok': 15535}

## Scope
Derived FAIR identifiers only. The frozen v3 analysis table, split manifest and feature cache are unchanged.

The v4 table adds canonical SMILES, InChI and InChIKey for machine-readable chemical identity. It does not replace the v3 frozen analysis input or alter previously reported features, splits or results.

RDKit emitted warnings for records with undefined stereochemistry. These records parsed successfully, but the derived identifiers do not restore or infer missing stereochemical assignments; downstream users should treat stereochemical completeness as an upstream-data limitation.

## Integrity
- Input SHA-256: `b6b0b6b75c9b8f1ca8efdefedea0ff133cf4847c68839a50f99dd6b02412603b`
- Output SHA-256: `dc6a81bfb075bb89b9bdc1e8c9db9fd22133af42d66edf020e268cf9f4127e8a`
- RDKit: `2026.03.5`
