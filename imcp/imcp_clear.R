mcp_curve<-function(y_true,y_score,abs_tolerance=1e-5) {
  y_true<-y_true$flatten()
  .validate_inputs(y_true,y_score,abs_tolerance)
  y_true_score<-torch::nnf_one_hot(y_true)[,1]
  curve_y<-.get_y_values(y_true,y_true_score,y_score)$curve_y
  n<-curve_y$numel()
  curve_x<-torch::torch_linspace(0,1,steps=n)
  list(curve_x=curve_x,curve_y=curve_y)
}

imcp_curve<-function(y_true,y_score,abs_tolerance=1e-5) {
  y_true<-y_true$flatten()
  .validate_inputs(y_true,y_score,abs_tolerance)
  y_true_score<-torch::nnf_one_hot(y_true)
  result<-.get_y_values(y_true,y_true_score,y_score)
  curve_y<-result$curve_y
  sort_indices<-result$sort_indices
  class_widths<-.get_class_widths(y_true_score)
  class_widths_per_sample<-class_widths[y_true]
  curve_x<-class_widths_per_sample[sort_indices]
  curve_x<-torch::torch_cumsum(curve_x,dim=1)-(curve_x/2)
  n_y<-curve_y$numel()
  dev <- y_score$device
  curve_x<-torch::torch_cat(
    list(torch::torch_zeros(1,device=dev),
    curve_x,torch::torch_ones(1,device=dev)))
  curve_y<-torch::torch_cat(list(
    curve_y[1]$unsqueeze(1),
    curve_y,
    curve_y[n_y]$unsqueeze(1)
  ))
  list(curve_x=curve_x,curve_y=curve_y)
}

mcp_score<-function(y_true,y_score,abs_tolerance=1e-5) {
  curves<-mcp_curve(y_true,y_score,abs_tolerance=abs_tolerance)
  as.numeric(torch::torch_trapz(curves$curve_y,x=curves$curve_x))
}

imcp_score<-function(y_true,y_score,abs_tolerance=1e-5) {
  curves<-imcp_curve(y_true,y_score,abs_tolerance=abs_tolerance)
  as.numeric(torch::torch_trapz(curves$curve_y,x=curves$curve_x))
}

.get_y_values<-function(y_true,y_true_score,y_score) {
  curve_y<-(y_true_score-torch::torch_sqrt(y_score))$pow(2)
  curve_y<-torch::torch_sum(curve_y,dim=2)
  curve_y<-torch::torch_sqrt(curve_y)/sqrt(2)
  curve_y<-1-curve_y
  secondary_indices<-torch::torch_sort(y_true, stable = TRUE)[[2]]
  curve_y_reordered<-curve_y[secondary_indices]
  primary_indices<-torch::torch_sort(curve_y_reordered,stable=TRUE)[[2]]
  sort_indices<-secondary_indices[primary_indices]
  curve_y<-curve_y[sort_indices]
  list(curve_y=curve_y,sort_indices=sort_indices)
}

.get_class_widths<-function(y_true_score) {
  class_widths<-torch::torch_sum(y_true_score,dim=1)
  1.0/(y_true_score$shape[2]*class_widths)
}

.validate_inputs<-function(y_true,y_score,abs_tolerance) {
  if(y_true$numel()!=y_score$shape[1]) {
    stop("'y_true' and 'y_score' have different number of samples")
  }
  row_sums<-y_score$sum(dim=2)
  if(!torch::torch_allclose(
    torch::torch_ones_like(row_sums),row_sums,rtol=0,atol=abs_tolerance)) {
    stop(
      "Target scores need to be probabilities, ",
      "i.e. they should sum up to 1.0 over classes"
    )
  }
}
