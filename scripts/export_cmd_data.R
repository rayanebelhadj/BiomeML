#!/usr/bin/env Rscript
# Export curatedMetagenomicData species abundance + metadata as TSVs that the
# BiomeML CMD loader (src/datasets/curated_metagenomic.py) can read.
#
# Output:
#   data/cmd/merged_abundance.tsv   features x samples (species rows, sample cols)
#   data/cmd/merged_metadata.tsv    samples x metadata fields (index = sample id)
#
# Run from the BiomeML repo root with the r-cmd conda env active:
#   ~/miniconda3/envs/r-cmd/bin/Rscript scripts/export_cmd_data.R

suppressPackageStartupMessages({
  library(curatedMetagenomicData)
  library(SummarizedExperiment)
})

OUT_DIR <- file.path(Sys.getenv("HOME"), "BiomeML", "data", "cmd")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("--- loading sampleMetadata ---\n")
md <- sampleMetadata
cat("Total CMD samples:", nrow(md), "\n")
cat("Columns:", paste(head(colnames(md), 12), collapse = ", "), "...\n")

# We want samples with a study_condition (the disease label column). Drop NA.
md_keep <- md[!is.na(md$study_condition) & nzchar(md$study_condition), ]
cat("Samples with study_condition:", nrow(md_keep), "\n")

cat("\nstudy_condition distribution (top 30):\n")
print(head(sort(table(md_keep$study_condition), decreasing = TRUE), 30))

# Keep only the columns the BiomeML loader uses, plus a few useful extras.
# The loader looks for: study_condition (disease), age, gender, BMI, body_site, study_name.
keep_cols <- intersect(
  c("sample_id", "study_name", "subject_id", "body_site", "antibiotics_current_use",
    "study_condition", "disease", "disease_subtype", "age", "infant_age",
    "age_category", "gender", "country", "BMI", "non_westernized", "DNA_extraction_kit",
    "PMID", "number_reads", "number_bases", "minimum_read_length", "median_read_length"),
  colnames(md_keep)
)
md_keep <- md_keep[, keep_cols, drop = FALSE]

# Fetch relative_abundance SummarizedExperiment for those samples.
# This pulls from ExperimentHub; the first call caches data locally (~hundreds of MB).
cat("\n--- fetching relative_abundance (may download from ExperimentHub) ---\n")
se <- returnSamples(
  sampleMetadata = md_keep,
  dataType       = "relative_abundance",
  rownames       = "long"
)
cat("SummarizedExperiment:", nrow(se), "features x", ncol(se), "samples\n")

# Write abundance: features as rows, samples as columns (CMD convention).
abund <- assay(se)
cat("\n--- writing", file.path(OUT_DIR, "merged_abundance.tsv"), "---\n")
write.table(abund,
            file      = file.path(OUT_DIR, "merged_abundance.tsv"),
            sep       = "\t",
            quote     = FALSE,
            col.names = NA)   # leaves the first column header empty so pandas can use it as the index

# Write metadata: samples as rows; pandas reads first column as the index, so sample_id
# must be the row name in R's view.
md_export <- as.data.frame(colData(se))
cat("--- writing", file.path(OUT_DIR, "merged_metadata.tsv"),
    "(", nrow(md_export), "rows x", ncol(md_export), "cols ) ---\n")
write.table(md_export,
            file      = file.path(OUT_DIR, "merged_metadata.tsv"),
            sep       = "\t",
            quote     = FALSE,
            col.names = NA)

cat("\nDone. Disease counts in exported metadata:\n")
print(head(sort(table(md_export$study_condition), decreasing = TRUE), 20))
