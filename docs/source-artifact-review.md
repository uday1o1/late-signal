# Source artifact review

## 2026-08-13 member-contract correction

The initial authored contract followed the source webpage's logical two-file description and expected flat members named `README.md` and `CriteoSearchData`.
An authorized download from the configured official Criteo Azure Blob URL completed at the predeclared byte count of 2,002,864,638 bytes but failed the exact member check.
No archive bytes were retained or trusted after that failure.

The safety inspector observed these members in this exact order:

1. `Criteo_Conversion_Search`
2. `Criteo_Conversion_Search/Criteo_Conversion_Search.tar.gz`
3. `Criteo_Conversion_Search/CriteoSearchData`
4. `Criteo_Conversion_Search/._README.md`
5. `Criteo_Conversion_Search/README.md`

The smallest correction is to lock these exact paths and point the raw schema inspector at the nested `Criteo_Conversion_Search/CriteoSearchData` member.
Archive traversal, link, special-file, member-count, member-size, total-expansion, compression-ratio, byte-count, and first-download digest gates remain unchanged.
This correction was made before any row was read and before any final-period result existed.
