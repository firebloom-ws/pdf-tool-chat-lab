void score_matrix(float* matrix, float* query, int rows, int cols, float* out_scores) {
  for (int row = 0; row < rows; row++) {
    float total = 0.0f;
    int base = row * cols;
    for (int col = 0; col < cols; col++) {
      total += matrix[base + col] * query[col];
    }
    out_scores[row] = total;
  }
}
