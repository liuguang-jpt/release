# Licence — Bias-Aware PROTAC Benchmark Release v1.0.0

This deposit contains project-authored software and documentation together with record-level material derived from third-party databases. The components do not share one blanket licence.

## 1. Project code: MIT License

The original code in `code/` and `external_code/` is licensed under the MIT License.

Copyright (c) 2026 Guanglu Liu and Shuang Wang, China University of Petroleum (East China)

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. Project-authored documentation: CC BY 4.0

Original narrative documentation, annotation protocols and reporting text authored for this project are licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0), except where a file or embedded item states otherwise.

Suggested attribution: Liu, G. & Wang, S. *Bias-Aware Learning from Censored and Incompletely Observed PROTAC Degradation Data: Dataset and Code*, version 1.0.0 (2026).

CC BY 4.0 does not override third-party rights in database-derived records, quoted material, trademarks or other upstream content.

## 3. PROTAC-DB-derived record-level material

The files under `data/derived/`, including the cleaned record-level table, annotation samples, split manifests and record identifiers, were derived from PROTAC-DB 3.0. They remain subject to the applicable PROTAC-DB access and reuse terms. This project **does not grant additional rights** over upstream database content and does not represent that all such files are CC BY 4.0.

The raw PROTAC-DB snapshot is not redistributed. `data/raw/raw_data_manifest.csv` is a provenance/hash manifest only. Users who need to rebuild the data should obtain an authorised source copy directly from PROTAC-DB and comply with its current terms.

## 4. TPDdb-derived external-evaluation material

TPDdb-derived record-level cohort tables, labels and per-record predictions are not included because the upstream redistribution licence was unresolved when this release candidate was prepared. `data/external/` contains aggregate metrics and a model manifest only; `external_code/` contains project-authored rebuild scripts licensed under MIT.

Users must obtain TPDdb directly under its applicable terms before running the external rebuild workflow. No licence to TPDdb content is granted by this deposit.

## 5. Other third-party sources

No PROTAC-PatentDB source records or other access-restricted third-party datasets are redistributed in this package. Citations and provenance statements identify upstream sources but do not alter their licences.

## 6. No warranty

All components are provided for research and reproducibility purposes without warranty. Users are responsible for verifying upstream terms, ethical constraints and journal or institutional requirements before reuse or redistribution.
