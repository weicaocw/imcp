# imcp.R — R port of imcp.py using torch operations instead of numpy
#
# Required packages: torch, ggplot2
# Install with:
#   install.packages("torch")
#   torch::install_torch()
#   install.packages("ggplot2")

library(torch)
library(ggplot2)


# ===========================================================================
# Internal helpers
# ===========================================================================

.map_class_labels <- function(y_true, y_score, labels) {

  unique_classes <- sort(unique(y_true))
  y_true_size   <- length(unique_classes)
  class_mapper  <- setNames(seq_len(y_true_size), unique_classes)
  y_true_int_encoded <- unname(class_mapper[as.character(y_true)])

  n_cols <- y_score$shape[2]

  if (y_true_size != n_cols) {
    if (is.null(labels)) {
      stop("Class labels not given!")
    }
    if (length(labels) != n_cols) {
      stop("Number of class labels not equal to the number of columns in 'y_score'")
    }
    if (!all(unique_classes %in% labels)) {
      stop(
        "Class labels from y_true are not a subset of given list of labels. ",
        "Check if values and types of given labels and y_true match."
      )
    }

    unique_classes <- sort(labels)
    y_true_size   <- length(unique_classes)
    class_mapper  <- setNames(seq_len(y_true_size), unique_classes)
    y_true_int_encoded <- unname(class_mapper[as.character(y_true)])
  }

  list(
    class_mapper       = class_mapper,
    y_true_size        = y_true_size,
    y_true_int_encoded = as.integer(y_true_int_encoded)
  )
}


.get_y_values <- function(y_true, y_true_score, y_score) {
  # Hellinger-distance-based y-values.
  # sqrt of y_true_score is omitted because it is a 0/1 one-hot matrix.
  curve_y <- (y_true_score - torch_sqrt(y_score))$pow(2)
  curve_y <- torch_sum(curve_y, dim = 2)
  curve_y <- torch_sqrt(curve_y) / sqrt(2)
  curve_y <- 1 - curve_y

  # lexsort: primary key = curve_y, secondary key = y_true
  curve_y_vec  <- as.numeric(curve_y)
  sort_indices <- order(curve_y_vec, y_true)

  curve_y <- curve_y[sort_indices]

  list(curve_y = curve_y, sort_indices = sort_indices)
}


.get_class_widths <- function(y_true_score, y_true_size) {
  # Width w_i = 1 / (n_classes * class_count_i)
  class_widths <- torch_sum(y_true_score, dim = 1)
  1.0 / (y_true_size * class_widths)
}


.torch_trapezoid <- function(y, x) {
  n     <- y$shape[1]
  dx    <- x[2:n] - x[1:(n - 1)]
  avg_y <- (y[2:n] + y[1:(n - 1)]) / 2.0
  torch_sum(dx * avg_y)
}


.get_color <- function(idx) {
  palette <- list(
    c(135, 181, 222),
    c(170, 122, 122),
    c(240, 128, 128),   # lightcoral
    c( 34, 139,  34),   # forestgreen
    c(238, 130, 238),   # violet
    c(220,  20,  60),   # crimson
    c(139,  69,  19),   # saddlebrown
    c(255, 140,   0),   # darkorange
    c( 65, 105, 225),   # royalblue
    c( 64, 224, 208),   # turquoise
    c( 95, 158, 160),   # cadetblue
    c( 30, 144, 255),   # dodgerblue
    c(138,  43, 226),   # blueviolet
    c(124, 252,   0),   # lawngreen
    c(255,  20, 147),   # deeppink
    c(  0, 255,   0),   # lime
    c(189, 183, 107),   # darkkhaki
    c(  0,   0,   0)
  )

  if (idx >= 1 && idx <= length(palette)) {
    v <- palette[[idx]] / 255
    rgb(v[1], v[2], v[3])
  } else {
    g <- (abs(idx) %% 256) / 255
    rgb(g, g, g)
  }
}


.ensure_score_tensor <- function(y_score) {
  if (is_torch_tensor(y_score)) y_score$to(dtype = torch_float())
  else torch_tensor(as.matrix(y_score), dtype = torch_float())
}


.validate_inputs <- function(y_true_vec, y_score_t, abs_tolerance) {
  if (length(y_true_vec) != y_score_t$shape[1]) {
    stop("'y_true' and 'y_score' have different number of samples")
  }

  row_sums <- y_score_t$sum(dim = 2)
  if (!torch_allclose(torch_ones_like(row_sums), row_sums, rtol = 0, atol = abs_tolerance)) {
    stop(
      "Target scores need to be probabilities, ",
      "i.e. they should sum up to 1.0 over classes"
    )
  }
}


# ===========================================================================
# Core curve / score functions
# ===========================================================================

#' Calculate MCP curve using Hellinger distance.
#'
#' @param y_true  Vector of true labels (length n_samples).
#' @param y_score Matrix or torch tensor of shape (n_samples, n_classes)
#'   with probability estimates.
#' @param labels  Optional vector of all class labels mapped to columns of
#'   \code{y_score}. Required when some classes are absent from
#'   \code{y_true}.
#' @param abs_tolerance Tolerance for checking that rows of \code{y_score}
#'   sum to 1.
#' @return A list with \code{curve_x} and \code{curve_y} (torch tensors).
mcp_curve <- function(y_true, y_score, labels = NULL, abs_tolerance = 1e-8) {

  y_true_vec <- as.vector(y_true)
  y_score_t  <- .ensure_score_tensor(y_score)
  .validate_inputs(y_true_vec, y_score_t, abs_tolerance)

  mapped         <- .map_class_labels(y_true_vec, y_score_t, labels)
  y_true_score   <- torch_eye(mapped$y_true_size)[mapped$y_true_int_encoded, ]

  result  <- .get_y_values(y_true_vec, y_true_score, y_score_t)
  curve_y <- result$curve_y

  n       <- curve_y$shape[1]
  curve_x <- torch_linspace(0, 1, steps = n)

  list(curve_x = curve_x, curve_y = curve_y)
}


#' Calculate imbalanced MCP curve using Hellinger distance.
#'
#' Unequal class distribution is taken into account.
#'
#' @inheritParams mcp_curve
#' @return A list with \code{curve_x} and \code{curve_y} (torch tensors).
imcp_curve <- function(y_true, y_score, labels = NULL, abs_tolerance = 1e-8) {

  y_true_vec <- as.vector(y_true)
  y_score_t  <- .ensure_score_tensor(y_score)
  .validate_inputs(y_true_vec, y_score_t, abs_tolerance)

  mapped             <- .map_class_labels(y_true_vec, y_score_t, labels)
  y_true_size        <- mapped$y_true_size
  y_true_int_encoded <- mapped$y_true_int_encoded

  y_true_score <- torch_eye(y_true_size)[y_true_int_encoded, ]

  result       <- .get_y_values(y_true_vec, y_true_score, y_score_t)
  curve_y      <- result$curve_y
  sort_indices <- result$sort_indices

  class_widths <- .get_class_widths(y_true_score, y_true_size)
  class_widths_per_sample <- class_widths[y_true_int_encoded]

  curve_x <- class_widths_per_sample[sort_indices]
  curve_x <- torch_cumsum(curve_x, dim = 1) - (curve_x / 2)

  # Prepend (0, y1) and append (1, yn)
  n_y     <- curve_y$shape[1]
  curve_x <- torch_cat(list(torch_zeros(1), curve_x, torch_ones(1)))
  curve_y <- torch_cat(list(
    curve_y[1]$unsqueeze(1),
    curve_y,
    curve_y[n_y]$unsqueeze(1)
  ))

  list(curve_x = curve_x, curve_y = curve_y)
}


#' Area under the MCP curve (trapezoid rule).
#'
#' @inheritParams mcp_curve
#' @return Scalar numeric area.
mcp_score <- function(y_true, y_score, labels = NULL, abs_tolerance = 1e-8) {
  curves <- mcp_curve(y_true, y_score, labels = labels, abs_tolerance = abs_tolerance)
  as.numeric(.torch_trapezoid(curves$curve_y, curves$curve_x))
}


#' Area under the imbalanced MCP curve (trapezoid rule).
#'
#' @inheritParams mcp_curve
#' @return Scalar numeric area.
imcp_score <- function(y_true, y_score, labels = NULL, abs_tolerance = 1e-8) {
  curves <- imcp_curve(y_true, y_score, labels = labels, abs_tolerance = abs_tolerance)
  as.numeric(.torch_trapezoid(curves$curve_y, curves$curve_x))
}


# ===========================================================================
# Plotting functions
# ===========================================================================

#' Plot one or more MCP curves.
#'
#' @param y_true  Vector of true labels.
#' @param y_score Matrix / torch tensor, **or** a named list of such objects
#'   (one per algorithm).
#' @inheritParams mcp_curve
#' @param output_fig_path Optional file path to save the figure.
plot_mcp_curve <- function(y_true, y_score, labels = NULL, abs_tolerance = 1e-8,
                           output_fig_path = NULL) {

  curves_list <- list()

  if (is.list(y_score) && !is_torch_tensor(y_score)) {
    for (key in names(y_score)) {
      area   <- round(mcp_score(y_true, y_score[[key]],
                                labels = labels, abs_tolerance = abs_tolerance), 4)
      curves <- mcp_curve(y_true, y_score[[key]],
                          labels = labels, abs_tolerance = abs_tolerance)
      curves_list[[length(curves_list) + 1]] <- list(
        x = as.numeric(curves$curve_x),
        y = as.numeric(curves$curve_y),
        label = sprintf("%s, AU(MCP)=%s", key, area)
      )
    }
  } else {
    area   <- round(mcp_score(y_true, y_score,
                              labels = labels, abs_tolerance = abs_tolerance), 4)
    curves <- mcp_curve(y_true, y_score,
                        labels = labels, abs_tolerance = abs_tolerance)
    curves_list[[1]] <- list(
      x = as.numeric(curves$curve_x),
      y = as.numeric(curves$curve_y),
      label = sprintf("clf1, AU(MCP)=%s", area)
    )
  }

  fig_title <- if (length(curves_list) > 1) "MCP curves" else "MCP curve"
  plot_curve(curves_list, output_fig_path = output_fig_path,
             fig_title = fig_title, ylabel = "MCP score")
}


#' Plot one or more imbalanced MCP curves.
#'
#' @inheritParams plot_mcp_curve
plot_imcp_curve <- function(y_true, y_score, labels = NULL, abs_tolerance = 1e-8,
                            output_fig_path = NULL) {

  curves_list <- list()

  if (is.list(y_score) && !is_torch_tensor(y_score)) {
    for (key in names(y_score)) {
      area   <- round(imcp_score(y_true, y_score[[key]],
                                 labels = labels, abs_tolerance = abs_tolerance), 4)
      curves <- imcp_curve(y_true, y_score[[key]],
                           labels = labels, abs_tolerance = abs_tolerance)
      curves_list[[length(curves_list) + 1]] <- list(
        x = as.numeric(curves$curve_x),
        y = as.numeric(curves$curve_y),
        label = sprintf("%s, AU(IMCP)=%s", key, area)
      )
    }
  } else {
    area   <- round(imcp_score(y_true, y_score,
                               labels = labels, abs_tolerance = abs_tolerance), 4)
    curves <- imcp_curve(y_true, y_score,
                         labels = labels, abs_tolerance = abs_tolerance)
    curves_list[[1]] <- list(
      x = as.numeric(curves$curve_x),
      y = as.numeric(curves$curve_y),
      label = sprintf("clf1, AU(IMCP)=%s", area)
    )
  }

  fig_title <- if (length(curves_list) > 1) "IMCP curves" else "IMCP curve"
  plot_curve(curves_list, output_fig_path = output_fig_path,
             fig_title = fig_title, ylabel = "IMCP score")
}


#' Generic curve plotter (ggplot2).
#'
#' @param curves_list A list of lists, each with elements \code{x}, \code{y},
#'   and \code{label} (all numeric vectors / character scalar).
#'   Alternatively, pass plain numeric vectors via \code{x} and \code{y}
#'   parameters for a single curve.
#' @param x Numeric vector (single curve) or list of numeric vectors.
#' @param y Numeric vector (single curve) or list of numeric vectors.
#' @param label Character scalar or vector of labels.
#' @param output_fig_path Optional path to save the figure.
#' @param fig_title Plot title.
#' @param xlabel X-axis label.
#' @param ylabel Y-axis label.
#' @return The ggplot object (invisibly).
plot_curve <- function(curves_list = NULL,
                       x = NULL, y = NULL, label = NULL,
                       output_fig_path = NULL,
                       fig_title = "(I)MCP curve(s)",
                       xlabel = "samples",
                       ylabel = "(I)MCP score") {

  # Accept either a pre-built curves_list or raw x / y / label arguments
  if (is.null(curves_list)) {
    if (is.null(x) || is.null(y)) stop("Provide either curves_list or x + y")

    # Normalise x / y to list-of-vectors
    if (is.numeric(x)) {
      x <- list(x)
      y <- list(y)
    }
    if (is.null(label)) label <- paste0("curve_", seq_along(x))
    if (is.character(label) && length(label) == 1) label <- list(label)

    curves_list <- mapply(function(xi, yi, li) list(x = xi, y = yi, label = li),
                          x, y, label, SIMPLIFY = FALSE)
  }

  df <- do.call(rbind, lapply(seq_along(curves_list), function(i) {
    data.frame(
      x     = curves_list[[i]]$x,
      y     = curves_list[[i]]$y,
      label = curves_list[[i]]$label,
      stringsAsFactors = FALSE
    )
  }))

  color_map <- setNames(
    sapply(seq_along(curves_list), .get_color),
    sapply(curves_list, `[[`, "label")
  )

  p <- ggplot(df, aes(x = .data$x, y = .data$y, color = .data$label)) +
    geom_line(linewidth = 0.8) +
    scale_color_manual(values = color_map) +
    scale_x_continuous(breaks = seq(0, 1, 0.1), limits = c(-0.005, 1.001)) +
    scale_y_continuous(breaks = seq(0, 1, 0.1), limits = c(-0.005, 1.001)) +
    coord_fixed() +
    labs(title = fig_title, x = xlabel, y = ylabel, color = NULL) +
    theme_minimal() +
    theme(
      panel.grid.major = element_line(
        color = "grey50", linewidth = 0.3, linetype = "dotted"
      ),
      panel.grid.minor = element_blank(),
      axis.text  = element_text(color = "grey50"),
      axis.title = element_text(color = "grey50"),
      plot.title = element_text(hjust = 0.5),
      legend.position = if (length(curves_list) > 0) "bottom" else "none"
    )

  if (!is.null(output_fig_path)) {
    ggsave(output_fig_path, plot = p, width = 9, height = 7)
  } else {
    print(p)
  }

  invisible(p)
}
