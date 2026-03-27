#!/usr/bin/env Rscript

required_packages <- c('ggplot2', 'dplyr', 'tidyr', 'patchwork')
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(
    sprintf(
      'Missing required R packages: %s',
      paste(missing_packages, collapse = ', ')
    ),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})

theme_set(
  theme_bw(base_size = 16) +
    theme(
      plot.title = element_text(size = 18, face = 'bold'),
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 14),
      legend.title = element_text(size = 14),
      legend.text = element_text(size = 13),
      strip.text = element_text(size = 14)
    )
)

script_path <- function() {
  command_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep('^--file=', command_args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub('^--file=', '', file_arg[[1]]), mustWork = FALSE))
  }
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(normalizePath(sys.frames()[[1]]$ofile, mustWork = FALSE))
  }
  normalizePath('test/plot_h5_benchmarks.R', mustWork = FALSE)
}

project_root <- function() {
  dirname(dirname(script_path()))
}

benchmark_results_dir <- function() {
  configured <- Sys.getenv('MODELARRAYIO_BENCHMARK_RESULTS_DIR', unset = '')
  if (nzchar(configured)) {
    return(normalizePath(path.expand(configured), mustWork = FALSE))
  }
  file.path(project_root(), 'benchmark_results')
}

default_results_csv_candidates <- function() {
  results_dir <- benchmark_results_dir()
  candidates <- sort(
    list.files(
      path = results_dir,
      pattern = '^h5_benchmark_results.*\\.csv$',
      full.names = TRUE
    )
  )
  if (length(candidates) > 0) {
    return(candidates)
  }
  file.path(results_dir, 'h5_benchmark_results.csv')
}

default_plot_dir <- function() {
  file.path(benchmark_results_dir(), 'plots')
}

default_readme_svg <- function() {
  file.path(project_root(), 'docs', '_static', 'h5_benchmark_summary.svg')
}

print_help <- function() {
  cat(
    paste(
      'Generate diagnostic plots for HDF5 benchmark runs (R + ggplot2).',
      '',
      'Usage:',
      '  Rscript test/plot_h5_benchmarks.R [options]',
      '',
      'Options:',
      '  --results-csv [PATH ...]   One or more benchmark CSV files. If omitted,',
      '                             auto-loads benchmark_results/h5_benchmark_results*.csv.',
      '  --output-dir PATH          Directory where plot images are written.',
      '  --readme-svg PATH          Path where README summary SVG is written.',
      '  --run-kind VALUE           One of: auto, all, quick, medium, full.',
      '  --sampled-voxels INT       Filter by sampled_voxels (-1 selects max available).',
      '  -h, --help                 Show this help and exit.',
      sep = '\n'
    )
  )
}

parse_cli_args <- function(args) {
  opts <- list(
    results_csv = NULL,
    output_dir = default_plot_dir(),
    readme_svg = default_readme_svg(),
    run_kind = 'auto',
    sampled_voxels = -1L
  )

  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c('-h', '--help')) {
      print_help()
      quit(save = 'no', status = 0L)
    }

    if (arg == '--results-csv') {
      i <- i + 1L
      values <- character()
      while (i <= length(args) && !startsWith(args[[i]], '--')) {
        values <- c(values, args[[i]])
        i <- i + 1L
      }
      opts$results_csv <- values
      next
    }

    if (arg == '--output-dir') {
      if (i >= length(args)) {
        stop('--output-dir requires a value', call. = FALSE)
      }
      i <- i + 1L
      opts$output_dir <- args[[i]]
      i <- i + 1L
      next
    }

    if (arg == '--readme-svg') {
      if (i >= length(args)) {
        stop('--readme-svg requires a value', call. = FALSE)
      }
      i <- i + 1L
      opts$readme_svg <- args[[i]]
      i <- i + 1L
      next
    }

    if (arg == '--run-kind') {
      if (i >= length(args)) {
        stop('--run-kind requires a value', call. = FALSE)
      }
      i <- i + 1L
      opts$run_kind <- args[[i]]
      i <- i + 1L
      next
    }

    if (arg == '--sampled-voxels') {
      if (i >= length(args)) {
        stop('--sampled-voxels requires a value', call. = FALSE)
      }
      i <- i + 1L
      opts$sampled_voxels <- suppressWarnings(as.integer(args[[i]]))
      if (is.na(opts$sampled_voxels)) {
        stop('--sampled-voxels must be an integer', call. = FALSE)
      }
      i <- i + 1L
      next
    }

    stop(sprintf('Unknown argument: %s', arg), call. = FALSE)
  }

  valid_run_kinds <- c('auto', 'all', 'quick', 'medium', 'full')
  if (!(opts$run_kind %in% valid_run_kinds)) {
    stop(
      sprintf(
        '--run-kind must be one of: %s',
        paste(valid_run_kinds, collapse = ', ')
      ),
      call. = FALSE
    )
  }

  opts
}

load_results <- function(paths) {
  if (length(paths) == 0) {
    stop('No benchmark results CSV paths were provided.', call. = FALSE)
  }

  data_frames <- list()
  missing_paths <- character()
  for (path in paths) {
    if (!file.exists(path)) {
      missing_paths <- c(missing_paths, path)
      next
    }
    data_frame <- read.csv(path, stringsAsFactors = FALSE)
    if (nrow(data_frame) > 0) {
      data_frame$results_csv_path <- path
      data_frames[[length(data_frames) + 1L]] <- data_frame
    }
  }

  if (length(data_frames) == 0) {
    if (length(missing_paths) > 0) {
      stop(
        sprintf(
          'benchmark results CSV not found: %s',
          paste(missing_paths, collapse = ', ')
        ),
        call. = FALSE
      )
    }
    stop('benchmark results CSV is empty', call. = FALSE)
  }

  data_frame <- bind_rows(data_frames)
  numeric_columns <- c(
    'num_input_files',
    'target_chunk_mb',
    'compression_level',
    'shuffle',
    'chunk_subjects',
    'chunk_items',
    'elapsed_seconds',
    'data_generation_seconds',
    'hdf5_write_seconds',
    'output_size_bytes',
    'output_size_gb',
    'throughput_values_per_second',
    'throughput_mb_per_second',
    'group_mask_voxels',
    'sampled_voxels',
    'mean_missing_fraction',
    'std_missing_fraction'
  )
  for (column in numeric_columns) {
    if (column %in% names(data_frame)) {
      data_frame[[column]] <- suppressWarnings(as.numeric(data_frame[[column]]))
    }
  }

  if (!('output_size_gb' %in% names(data_frame))) {
    data_frame$output_size_gb <- data_frame$output_size_bytes / (1024.0^3)
  }
  if ('shuffle' %in% names(data_frame)) {
    data_frame$shuffle_label <- ifelse(
      data_frame$shuffle == 1,
      'on',
      ifelse(data_frame$shuffle == 0, 'off', 'unknown')
    )
  }

  filtered <- data_frame |>
    filter(
      !is.na(num_input_files),
      !is.na(target_chunk_mb),
      !is.na(elapsed_seconds),
      !is.na(output_size_bytes)
    )
  if (nrow(filtered) == 0) {
    stop(
      'No benchmark rows contained required columns after CSV loading.',
      call. = FALSE
    )
  }
  filtered
}

auto_run_kind <- function(data_frame) {
  if (!('run_kind' %in% names(data_frame))) {
    return(NULL)
  }
  run_kinds <- sort(unique(as.character(data_frame$run_kind[!is.na(data_frame$run_kind)])))
  for (preferred in c('full', 'medium', 'quick')) {
    if (preferred %in% run_kinds) {
      return(preferred)
    }
  }
  if (length(run_kinds) > 0) {
    return(run_kinds[[length(run_kinds)]])
  }
  NULL
}

filter_comparable_results <- function(data_frame, run_kind, sampled_voxels) {
  filtered <- data_frame
  details <- character()

  if (run_kind != 'all' && 'run_kind' %in% names(filtered)) {
    selected_run_kind <- if (run_kind == 'auto') {
      auto_run_kind(filtered)
    } else {
      run_kind
    }
    if (!is.null(selected_run_kind)) {
      subset <- filtered[filtered$run_kind == selected_run_kind, , drop = FALSE]
      if (nrow(subset) == 0) {
        if (run_kind != 'auto') {
          stop(
            sprintf('No rows found for requested run_kind=%s', run_kind),
            call. = FALSE
          )
        }
      } else {
        filtered <- subset
        details <- c(details, sprintf('run_kind=%s', selected_run_kind))
      }
    }
  }

  if ('sampled_voxels' %in% names(filtered) && any(!is.na(filtered$sampled_voxels))) {
    target_sampled_voxels <- if (sampled_voxels < 0) {
      as.integer(max(filtered$sampled_voxels, na.rm = TRUE))
    } else {
      as.integer(sampled_voxels)
    }
    subset <- filtered[filtered$sampled_voxels == target_sampled_voxels, , drop = FALSE]
    if (nrow(subset) == 0) {
      if (sampled_voxels >= 0) {
        stop(
          sprintf(
            'No rows found for requested sampled_voxels=%s',
            target_sampled_voxels
          ),
          call. = FALSE
        )
      }
    } else {
      filtered <- subset
      details <- c(details, sprintf('sampled_voxels=%s', target_sampled_voxels))
    }
  }

  if (nrow(filtered) == 0) {
    stop(
      'No benchmark rows left after filtering for comparable runs.',
      call. = FALSE
    )
  }

  label <- if (length(details) > 0) {
    paste(details, collapse = ', ')
  } else {
    'all available rows'
  }
  list(data = filtered, label = label)
}

annotate_compression_fields <- function(data_frame) {
  if (!('compression' %in% names(data_frame))) {
    stop("CSV must contain a 'compression' column.", call. = FALSE)
  }
  if (!('compression_level' %in% names(data_frame))) {
    data_frame$compression_level <- NA_real_
  }

  data_frame <- data_frame |>
    mutate(
      compression_program = as.character(compression),
      compression_level_num = suppressWarnings(as.integer(compression_level)),
      compression_variant = ifelse(
        compression_program == 'gzip' & !is.na(compression_level_num),
        paste0('gzip-', compression_level_num),
        compression_program
      )
    )

  preferred_program_order <- c('none', 'lzf', 'gzip')
  seen_programs <- sort(unique(data_frame$compression_program))
  program_levels <- c(
    preferred_program_order[preferred_program_order %in% seen_programs],
    sort(setdiff(seen_programs, preferred_program_order))
  )

  variant_table <- data_frame |>
    distinct(compression_program, compression_variant, compression_level_num) |>
    mutate(
      program_rank = match(compression_program, program_levels),
      level_rank = ifelse(compression_program == 'gzip', compression_level_num, -1L)
    ) |>
    arrange(program_rank, level_rank, compression_variant)
  variant_levels <- unique(variant_table$compression_variant)

  data_frame$compression_program <- factor(
    data_frame$compression_program,
    levels = program_levels
  )
  data_frame$compression_variant <- factor(
    data_frame$compression_variant,
    levels = variant_levels
  )

  list(
    data = data_frame,
    program_levels = program_levels,
    variant_levels = variant_levels
  )
}

program_palette <- function(program_levels) {
  fixed <- c(none = '#2ca02c', lzf = '#ff7f0e', gzip = '#1f77b4')
  palette <- fixed[names(fixed) %in% program_levels]
  unknown_levels <- setdiff(program_levels, names(palette))
  if (length(unknown_levels) > 0) {
    palette <- c(palette, stats::setNames(rep('#7f7f7f', length(unknown_levels)), unknown_levels))
  }
  palette[program_levels]
}

variant_shape_map <- function(variant_levels) {
  base_shapes <- c(16, 17, 15, 18, 3, 7, 8, 9, 10, 12, 13, 14, 0, 1, 2, 4, 5, 6, 11)
  stats::setNames(rep(base_shapes, length.out = length(variant_levels)), variant_levels)
}

apply_compression_scales <- function(plot, program_colors, shape_values) {
  plot +
    scale_color_manual(
      values = program_colors,
      name = 'Compression program',
      drop = FALSE
    ) +
    scale_shape_manual(
      values = shape_values,
      name = 'Compression variant',
      drop = FALSE
    )
}

build_line_metric_plot <- function(
  data_frame,
  metric,
  y_label,
  title,
  program_colors,
  shape_values
) {
  grouped <- data_frame |>
    group_by(num_input_files, compression_program, compression_variant) |>
    summarise(metric_value = median(.data[[metric]], na.rm = TRUE), .groups = 'drop') |>
    arrange(num_input_files)

  plot <- ggplot(
    grouped,
    aes(
      x = num_input_files,
      y = metric_value,
      color = compression_program,
      shape = compression_variant,
      group = compression_variant
    )
  ) +
    geom_line(linewidth = 0.7, show.legend = FALSE) +
    geom_point(size = 2.2) +
    scale_x_log10() +
    labs(
      title = title,
      x = 'Number of input files',
      y = y_label
    ) +
    theme(legend.position = 'bottom')

  apply_compression_scales(plot, program_colors, shape_values)
}

build_pareto_plot <- function(data_frame, program_colors, shape_values) {
  plot <- ggplot(
    data_frame,
    aes(
      x = elapsed_seconds,
      y = output_size_gb,
      color = compression_program,
      shape = compression_variant
    )
  ) +
    geom_point(alpha = 0.65, size = 2.2) +
    labs(
      title = 'Pareto view: write time vs output size by compression variant (all runs)',
      x = 'Write time (seconds)',
      y = 'Output size (GiB)'
    ) +
    theme(legend.position = 'bottom')

  apply_compression_scales(plot, program_colors, shape_values)
}

build_chunk_geometry_plot <- function(data_frame) {
  grouped <- data_frame |>
    group_by(num_input_files, target_chunk_mb) |>
    summarise(
      chunk_items = median(chunk_items, na.rm = TRUE),
      chunk_subjects = median(chunk_subjects, na.rm = TRUE),
      .groups = 'drop'
    ) |>
    pivot_longer(
      cols = c(chunk_items, chunk_subjects),
      names_to = 'metric',
      values_to = 'metric_value'
    )

  label_map <- c(
    chunk_items = 'Chunk items vs target chunk size',
    chunk_subjects = 'Chunk subjects vs target chunk size'
  )
  grouped$metric_label <- factor(grouped$metric, levels = names(label_map), labels = label_map)

  ggplot(
    grouped,
    aes(
      x = target_chunk_mb,
      y = metric_value,
      color = factor(num_input_files),
      group = factor(num_input_files)
    )
  ) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 2) +
    facet_wrap(~metric_label, nrow = 1, scales = 'free_y') +
    labs(
      title = 'Chunk size across cohort sizes',
      x = 'Target chunk size (MiB)',
      y = 'Median chunk size (count)',
      color = 'num_input_files'
    ) +
    theme(legend.position = 'bottom')
}

build_chunk_tradeoff_plot <- function(data_frame, program_colors, shape_values) {
  grouped <- data_frame |>
    group_by(compression_program, compression_variant, target_chunk_mb) |>
    summarise(
      elapsed_seconds = median(elapsed_seconds, na.rm = TRUE),
      output_size_gb = median(output_size_gb, na.rm = TRUE),
      .groups = 'drop'
    ) |>
    pivot_longer(
      cols = c(elapsed_seconds, output_size_gb),
      names_to = 'metric',
      values_to = 'metric_value'
    )

  label_map <- c(
    elapsed_seconds = 'Write time vs chunk target (seconds)',
    output_size_gb = 'Output size vs chunk target (GiB)'
  )
  grouped$metric_label <- factor(grouped$metric, levels = names(label_map), labels = label_map)

  plot <- ggplot(
    grouped,
    aes(
      x = target_chunk_mb,
      y = metric_value,
      color = compression_program,
      shape = compression_variant,
      group = compression_variant
    )
  ) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 2.2) +
    facet_wrap(~metric_label, nrow = 1, scales = 'free_y') +
    labs(
      title = 'Chunk target trade-off by compression variant',
      x = 'Target chunk size (MiB)',
      y = NULL
    ) +
    theme(
      legend.position = 'bottom',
      axis.title.y = element_blank()
    )

  apply_compression_scales(plot, program_colors, shape_values)
}

build_gzip_level_plot <- function(data_frame) {
  gzip_frame <- data_frame |>
    filter(compression_program == 'gzip')
  if (nrow(gzip_frame) == 0) {
    return(NULL)
  }

  grouped <- gzip_frame |>
    group_by(compression_level_num, target_chunk_mb) |>
    summarise(
      elapsed_seconds = median(elapsed_seconds, na.rm = TRUE),
      output_size_gb = median(output_size_gb, na.rm = TRUE),
      .groups = 'drop'
    ) |>
    mutate(compression_level_label = paste0('gzip-', compression_level_num)) |>
    pivot_longer(
      cols = c(elapsed_seconds, output_size_gb),
      names_to = 'metric',
      values_to = 'metric_value'
    )

  label_map <- c(
    elapsed_seconds = 'Gzip level effect on write time (seconds)',
    output_size_gb = 'Gzip level effect on output size (GiB)'
  )
  grouped$metric_label <- factor(grouped$metric, levels = names(label_map), labels = label_map)

  ggplot(
    grouped,
    aes(
      x = target_chunk_mb,
      y = metric_value,
      color = compression_level_label,
      shape = compression_level_label,
      group = compression_level_label
    )
  ) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 2.2) +
    facet_wrap(~metric_label, nrow = 1, scales = 'free_y') +
    labs(
      title = 'Gzip level trade-off by chunk size',
      x = 'Target chunk size (MiB)',
      y = NULL,
      color = 'compression_level',
      shape = 'compression_level'
    ) +
    theme(
      legend.position = 'bottom',
      axis.title.y = element_blank()
    )
}

build_shuffle_effect_plot <- function(data_frame, program_colors, shape_values) {
  if (!('shuffle_label' %in% names(data_frame))) {
    return(NULL)
  }

  grouped <- data_frame |>
    filter(shuffle_label %in% c('on', 'off')) |>
    group_by(compression_program, compression_variant, shuffle_label) |>
    summarise(
      elapsed_seconds = median(elapsed_seconds, na.rm = TRUE),
      output_size_gb = median(output_size_gb, na.rm = TRUE),
      throughput_mb_per_second = median(throughput_mb_per_second, na.rm = TRUE),
      .groups = 'drop'
    )
  if (nrow(grouped) == 0) {
    return(NULL)
  }

  grouped$shuffle_label <- factor(grouped$shuffle_label, levels = c('off', 'on'))
  metric_labels <- c(
    elapsed_seconds = 'Write time (seconds)',
    output_size_gb = 'Output size (GiB)',
    throughput_mb_per_second = 'Throughput (MiB/sec)'
  )
  long_grouped <- grouped |>
    pivot_longer(
      cols = c(elapsed_seconds, output_size_gb, throughput_mb_per_second),
      names_to = 'metric',
      values_to = 'metric_value'
    )
  long_grouped$metric_label <- factor(
    long_grouped$metric,
    levels = names(metric_labels),
    labels = metric_labels
  )

  plot <- ggplot(
    long_grouped,
    aes(
      x = shuffle_label,
      y = metric_value,
      color = compression_program,
      shape = compression_variant,
      group = compression_variant
    )
  ) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 2.2) +
    facet_wrap(~metric_label, nrow = 1, scales = 'free_y') +
    labs(
      title = 'Shuffle effect by compression variant (median across all benchmark rows)',
      x = 'Shuffle',
      y = NULL
    ) +
    theme(
      legend.position = 'bottom',
      axis.title.y = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.line.y = element_blank()
    )

  apply_compression_scales(plot, program_colors, shape_values)
}

save_plot_svg <- function(plot, path, width, height) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  grDevices::svg(filename = path, width = width, height = height)
  print(plot)
  grDevices::dev.off()
}

main <- function() {
  args <- parse_cli_args(commandArgs(trailingOnly = TRUE))
  selected_results_csv <- args$results_csv
  if (is.null(selected_results_csv) || length(selected_results_csv) == 0) {
    selected_results_csv <- default_results_csv_candidates()
  }

  data_frame <- load_results(selected_results_csv)
  filtered <- filter_comparable_results(
    data_frame = data_frame,
    run_kind = args$run_kind,
    sampled_voxels = args$sampled_voxels
  )
  data_frame <- filtered$data
  selection_label <- filtered$label

  compression_info <- annotate_compression_fields(data_frame)
  data_frame <- compression_info$data
  program_colors <- program_palette(compression_info$program_levels)
  shape_values <- variant_shape_map(compression_info$variant_levels)

  p_scaling_time <- build_line_metric_plot(
    data_frame = data_frame,
    metric = 'elapsed_seconds',
    y_label = 'Write time (seconds)',
    title = 'Median write time vs number of input files',
    program_colors = program_colors,
    shape_values = shape_values
  )
  p_scaling_size <- build_line_metric_plot(
    data_frame = data_frame,
    metric = 'output_size_gb',
    y_label = 'Output size (GiB)',
    title = 'Median output size vs number of input files',
    program_colors = program_colors,
    shape_values = shape_values
  )
  p_scaling_tput_values <- build_line_metric_plot(
    data_frame = data_frame,
    metric = 'throughput_values_per_second',
    y_label = 'Throughput (values/sec)',
    title = 'Median throughput (values/sec) vs number of input files',
    program_colors = program_colors,
    shape_values = shape_values
  )
  p_scaling_tput_mib <- build_line_metric_plot(
    data_frame = data_frame,
    metric = 'throughput_mb_per_second',
    y_label = 'Throughput (MiB/sec)',
    title = 'Median throughput (MiB/sec) vs number of input files',
    program_colors = program_colors,
    shape_values = shape_values
  )
  p_pareto <- build_pareto_plot(
    data_frame = data_frame,
    program_colors = program_colors,
    shape_values = shape_values
  )
  p_chunk_geometry <- build_chunk_geometry_plot(data_frame)
  p_chunk_tradeoff <- build_chunk_tradeoff_plot(
    data_frame = data_frame,
    program_colors = program_colors,
    shape_values = shape_values
  )
  p_gzip_level <- build_gzip_level_plot(data_frame)
  p_shuffle <- build_shuffle_effect_plot(
    data_frame = data_frame,
    program_colors = program_colors,
    shape_values = shape_values
  )

  p_scaling_size_summary <- p_scaling_size +
    guides(color = 'none', shape = 'none')
  p_scaling_tput_summary <- p_scaling_tput_mib +
    labs(title = 'Median throughput (MiB/sec)\nvs number of input files') +
    guides(color = 'none', shape = 'none')
  p_pareto_summary <- p_pareto +
    labs(title = 'Pareto view:\nwrite time vs output size (all runs)') +
    guides(color = 'none', shape = 'none')
  p_shuffle_summary <- p_shuffle +
    labs(title = 'Shuffle effect by compression variant') +
    guides(color = 'none', shape = 'none')

  summary_plot <- (
    (p_scaling_time + p_scaling_size_summary) /
    (p_scaling_tput_summary + p_pareto_summary) /
    p_shuffle_summary
  ) +
    plot_layout(guides = 'collect') +
    plot_annotation(
      title = sprintf('HDF5 benchmark summary (%s)', selection_label)
    ) &
    theme(
      legend.position = 'bottom',
      legend.justification = 'center',
      legend.direction = 'horizontal',
      legend.box = 'horizontal',
      legend.box.just = 'center',
      legend.key.size = grid::unit(0.8, 'lines'),
      legend.spacing.y = grid::unit(0.15, 'cm'),
      plot.margin = margin(t = 8, r = 8, b = 16, l = 8)
    )

  output_dir <- normalizePath(args$output_dir, mustWork = FALSE)
  readme_svg <- normalizePath(args$readme_svg, mustWork = FALSE)

  save_plot_svg(
    plot = summary_plot,
    path = file.path(output_dir, 'h5_benchmark_summary.svg'),
    width = 13,
    height = 13
  )
  save_plot_svg(
    plot = summary_plot,
    path = readme_svg,
    width = 13,
    height = 13
  )
  save_plot_svg(
    plot = p_scaling_time,
    path = file.path(output_dir, 'scaling_time_vs_inputs.svg'),
    width = 8,
    height = 5
  )
  save_plot_svg(
    plot = p_scaling_size,
    path = file.path(output_dir, 'scaling_size_vs_inputs.svg'),
    width = 8,
    height = 5
  )
  save_plot_svg(
    plot = p_scaling_tput_values,
    path = file.path(output_dir, 'scaling_throughput_values_vs_inputs.svg'),
    width = 8,
    height = 5
  )
  save_plot_svg(
    plot = p_scaling_tput_mib,
    path = file.path(output_dir, 'scaling_throughput_mib_vs_inputs.svg'),
    width = 8,
    height = 5
  )
  save_plot_svg(
    plot = p_pareto,
    path = file.path(output_dir, 'pareto_size_vs_time.svg'),
    width = 8,
    height = 6
  )
  save_plot_svg(
    plot = p_chunk_geometry,
    path = file.path(output_dir, 'chunk_geometry_vs_target_chunk.svg'),
    width = 12,
    height = 4.5
  )
  save_plot_svg(
    plot = p_chunk_tradeoff,
    path = file.path(output_dir, 'chunk_tradeoff_time_and_size.svg'),
    width = 12,
    height = 4.5
  )
  if (!is.null(p_gzip_level)) {
    save_plot_svg(
      plot = p_gzip_level,
      path = file.path(output_dir, 'gzip_level_tradeoff_time_and_size.svg'),
      width = 12,
      height = 4.5
    )
  }
  if (!is.null(p_shuffle)) {
    save_plot_svg(
      plot = p_shuffle,
      path = file.path(output_dir, 'shuffle_effect_summary.svg'),
      width = 15,
      height = 4.5
    )
  }

  cat(sprintf('Wrote SVG plots to %s\n', output_dir))
  cat(sprintf('Updated README SVG summary at %s\n', readme_svg))
  cat(sprintf('Loaded CSV files: %s\n', paste(selected_results_csv, collapse = ', ')))
  cat(sprintf('Plot row filter: %s\n', selection_label))
  0L
}

status <- main()
quit(save = 'no', status = status)
