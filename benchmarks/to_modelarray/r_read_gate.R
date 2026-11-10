#!/usr/bin/env Rscript

user_lib <- Sys.getenv("MODELARRAYIO_R_LIBS", "")
if (nzchar(user_lib) && dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: r_read_gate.R OUTPUT BACKEND")
}

output <- args[[1L]]
backend <- args[[2L]]
started <- proc.time()[["elapsed"]]

if (backend == "hdf5") {
  if (!requireNamespace("rhdf5", quietly = TRUE) ||
      !requireNamespace("HDF5Array", quietly = TRUE) ||
      !requireNamespace("ModelArray", quietly = TRUE)) {
    stop("The HDF5 gate requires rhdf5, HDF5Array, and ModelArray")
  }
  listing <- rhdf5::h5ls(output, recursive = TRUE)
  scalar_rows <- listing$name == "values" & grepl("^/scalars/[^/]+$", listing$group)
  scalar_names <- sub("^/scalars/", "", listing$group[scalar_rows])
  if (!length(scalar_names)) stop("No scalar matrices found")
  modelarray <- ModelArray::ModelArray(output, scalar_types = scalar_names)
  scalar <- ModelArray::scalars(modelarray)[[scalar_names[[1L]]]]
  invisible(as.matrix(scalar[seq_len(min(64L, nrow(scalar))),
                             seq_len(min(8L, ncol(scalar))), drop = FALSE]))
} else if (backend == "tiledb") {
  if (!requireNamespace("tiledb", quietly = TRUE)) {
    stop("The TileDB gate requires the tiledb R package")
  }
  scalar_roots <- list.dirs(file.path(output, "scalars"), recursive = FALSE)
  value_uris <- file.path(scalar_roots, "values")
  value_uris <- value_uris[file.exists(value_uris)]
  if (!length(value_uris)) stop("No TileDB scalar matrices found")
  array <- tiledb::tiledb_array(value_uris[[1L]], query_type = "READ")
  invisible(array[0L, 0L])
} else {
  stop(sprintf("Unknown backend: %s", backend))
}

elapsed <- proc.time()[["elapsed"]] - started
cat(sprintf("read_seconds=%.6f\n", elapsed))
