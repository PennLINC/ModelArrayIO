#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
results_root <- if (length(args) >= 1L) args[[1L]] else Sys.getenv(
  "MODELARRAYIO_BENCHMARK_RESULTS_DIR", "benchmarks/to_modelarray/results"
)
plot_dir <- if (length(args) >= 2L) args[[2L]] else file.path(results_root, "plots")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

files <- list.files(results_root, pattern = "^results(?:\\.[^.]+)?\\.csv$",
                    recursive = TRUE, full.names = TRUE)
if (!length(files)) stop("No benchmark result CSVs found")

tables <- lapply(files, function(path) read.csv(path, stringsAsFactors = FALSE))
all_names <- unique(unlist(lapply(tables, names)))
tables <- lapply(tables, function(table) {
  for (name in setdiff(all_names, names(table))) table[[name]] <- NA
  table[all_names]
})
results <- do.call(rbind, tables)
results <- results[results$status == "passed" & results$run_kind == "end_to_end", , drop = FALSE]
if ("r_gate" %in% names(results)) {
  gate_failed <- !is.na(results$r_gate) & grepl("^failed", results$r_gate)
  results <- results[!gate_failed, , drop = FALSE]
}
if (!nrow(results)) stop("No passing end-to-end benchmark rows found")

png(file.path(plot_dir, "runtime_by_workers.png"), width = 1200, height = 800, res = 130)
groups <- interaction(results$backend, results$dtype, results$compression, drop = TRUE)
colours <- seq_along(levels(groups))
plot(results$workers, results$wall_seconds, log = "xy", pch = 19,
     col = colours[groups], xlab = "Loader workers", ylab = "Wall time (seconds)",
     main = "to-modelarray end-to-end runtime")
legend("topright", legend = levels(groups), col = colours, pch = 19, cex = 0.8)
dev.off()

phase_names <- intersect(
  c("source_load_seconds", "validation_seconds", "staging_allocation_seconds",
    "compression_seconds", "submission_seconds"),
  names(results)
)
if (length(phase_names)) {
  phase_values <- sapply(phase_names, function(name) {
    values <- suppressWarnings(as.numeric(results[[name]]))
    mean(values, na.rm = TRUE)
  })
  phase_values[!is.finite(phase_values)] <- 0
  png(file.path(plot_dir, "mean_phase_seconds.png"), width = 1200, height = 800, res = 130)
  barplot(phase_values, las = 2, ylab = "Mean seconds", main = "Mean recorded phase time")
  dev.off()
}
