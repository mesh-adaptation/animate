/*
 * C kernels for 2D metrics.
 *
 * All matrices are stored row-major; metrics are 2x2 symmetric positive
 * definite. The eigenvalue solver is the closed-form Jacobi rotation for a
 * symmetric 2x2 matrix (stable even when the off-diagonal is tiny compared
 * with the diagonal). Eigenpairs are returned in *ascending* eigenvalue
 * order, matching the ordering convention of LAPACK dsyev.
 *
 * This file is consumed by animate._kernel.get_metric_kernel, which parses
 * the PREAMBLE section (all static helpers) and each public function body
 * separately, emitting them as a loopy kernel via op3.Function.from_c_string.
 * Preamble markers use a comment convention the parser knows:
 *
 *     // PREAMBLE-BEGIN
 *     ... helpers ...
 *     // PREAMBLE-END
 *
 * Public kernels then appear as plain C functions whose body is what we
 * want loopy to inline.
 */

/* PREAMBLE-BEGIN */

#include <math.h>

/*
 * Solve the eigenvalue problem for a symmetric 2x2 row-major matrix,
 * storing eigenvectors as columns of Q and eigenvalues in D (ascending).
 */
static void eig2x2(double Q[4], double D[2], const double A[4]) {
  const double a = A[0];
  const double b = A[1];
  const double c = A[3];

  if (abs(b) < 1e-16) {
    if (a <= c) {
      Q[0] = 1.0;
      Q[1] = 0.0;
      Q[2] = 0.0;
      Q[3] = 1.0;
      D[0] = a;
      D[1] = c;
    } else {
      Q[0] = 0.0;
      Q[1] = 1.0;
      Q[2] = 1.0;
      Q[3] = 0.0;
      D[0] = c;
      D[1] = a;
    }
    return;
  }

  const double m = 0.5 * (a + c);
  const double d = sqrt(0.25 * (a - c) * (a - c) + b * b);
  const double l1 = m - d; /* smaller */
  const double l2 = m + d; /* larger  */

  /* Eigenvector for l1: a nonzero null vector of M - l1 I. Pick the more
   * numerically robust of (b, l1 - a) and (l1 - c, b).
   */
  const double e1x = b;
  const double e1y = l1 - a;
  const double f1x = l1 - c;
  const double f1y = b;
  const double ne2 = e1x * e1x + e1y * e1y;
  const double nf2 = f1x * f1x + f1y * f1y;
  const double vx = (ne2 >= nf2) ? e1x : f1x;
  const double vy = (ne2 >= nf2) ? e1y : f1y;
  const double n = sqrt(vx * vx + vy * vy);
  const double ux = vx / n;
  const double uy = vy / n;

  /* Second eigenvector is the unit vector perpendicular to (ux, uy). */
  Q[0] = ux;
  Q[1] = -uy;
  Q[2] = uy;
  Q[3] = ux;
  D[0] = l1;
  D[1] = l2;
}

/*
 * Form A = Q diag(d0, d1) Q^T for a 2x2 row-major orthogonal Q.
 */
static void rqdt_2x2(double M[4], const double Q[4], double d0, double d1) {
  const double q00 = Q[0], q01 = Q[1];
  const double q10 = Q[2], q11 = Q[3];
  M[0] = d0 * q00 * q00 + d1 * q01 * q01;
  M[1] = d0 * q00 * q10 + d1 * q01 * q11;
  M[2] = d0 * q10 * q00 + d1 * q11 * q01;
  M[3] = d0 * q10 * q10 + d1 * q11 * q11;
}

/* PREAMBLE-END */

/*
 * Extract eigenvectors and eigenvalues from a 2D metric field.
 */
void get_eigendecomposition(double EVecs_[4], double EVals_[2],
                            const double *M_) {
  eig2x2(EVecs_, EVals_, M_);
}

/*
 * Extract eigenvectors and eigenvalues from a 2D metric field, with
 * eigenvalues **decreasing** in magnitude.
 */
void get_reordered_eigendecomposition(double EVecs_[4], double EVals_[2],
                                      const double *M_) {
  double D[2];
  double Q[4];
  eig2x2(Q, D, M_);
  if (fabs(D[0]) >= fabs(D[1])) {
    EVecs_[0] = Q[0];
    EVecs_[1] = Q[1];
    EVecs_[2] = Q[2];
    EVecs_[3] = Q[3];
    EVals_[0] = D[0];
    EVals_[1] = D[1];
  } else {
    /* Swap columns to put the larger-magnitude eigenpair first. */
    EVecs_[0] = Q[1];
    EVecs_[1] = Q[0];
    EVecs_[2] = Q[3];
    EVecs_[3] = Q[2];
    EVals_[0] = D[1];
    EVals_[1] = D[0];
  }
}

/*
 * Post-process a 2D metric field in order to enforce maximum and minimum
 * tolerated metric magnitudes, as well as maximum tolerated anisotropy.
 */
void postproc_metric(double A_[4], const double *h_min, const double *h_max,
                     const double *a_max) {
  const double l_max = pow(*h_min, -2);
  const double l_min = pow(*h_max, -2);
  const double la_min = pow(*a_max, -2);

  double Q[4];
  double D[2];
  eig2x2(Q, D, A_);

  /* Enforce maximum and minimum tolerated metric magnitudes. */
  double d0 = fmin(l_max, fmax(l_min, fabs(D[0])));
  double d1 = fmin(l_max, fmax(l_min, fabs(D[1])));
  const double max_eig = fmax(d0, d1);

  /* Enforce maximum tolerated anisotropy. */
  d0 = fmax(d0, la_min * max_eig);
  d1 = fmax(d1, la_min * max_eig);

  /* Reassemble the metric with the clamped eigenvalues. */
  rqdt_2x2(A_, Q, d0, d1);
}

/*
 * Intersect two 2D metric fields.
 */
void intersect(double M_[4], const double *A_, const double *B_) {
  /* Eigendecomposition of A: A = Qa Da Qa^T. */
  double Qa[4];
  double Da[2];
  eig2x2(Qa, Da, A_);

  /* Square root and inverse square root of A. */
  double Sq[4];
  double Sqi[4];
  rqdt_2x2(Sq, Qa, sqrt(Da[0]), sqrt(Da[1]));
  rqdt_2x2(Sqi, Qa, 1.0 / sqrt(Da[0]), 1.0 / sqrt(Da[1]));

  /* C = Sqi^T B Sqi (Sqi symmetric). */
  const double b00 = B_[0], b01 = B_[1];
  const double b10 = B_[2], b11 = B_[3];
  const double s00 = Sqi[0], s01 = Sqi[1];
  const double s10 = Sqi[2], s11 = Sqi[3];
  /* tmp = B * Sqi */
  const double t00 = b00 * s00 + b01 * s10;
  const double t01 = b00 * s01 + b01 * s11;
  const double t10 = b10 * s00 + b11 * s10;
  const double t11 = b10 * s01 + b11 * s11;
  /* C = Sqi * tmp (Sqi symmetric, so Sqi = Sqi^T). */
  double C[4];
  C[0] = s00 * t00 + s01 * t10;
  C[1] = s00 * t01 + s01 * t11;
  C[2] = s10 * t00 + s11 * t10;
  C[3] = s10 * t01 + s11 * t11;

  /* Eigendecomposition of C = Qc Dc Qc^T; clamp eigenvalues below at 1. */
  double Qc[4];
  double Dc[2];
  eig2x2(Qc, Dc, C);
  const double e0 = fmax(Dc[0], 1.0);
  const double e1 = fmax(Dc[1], 1.0);

  /* M = Sq^T (Qc diag(e) Qc^T) Sq; both factors symmetric. */
  double QC[4];
  rqdt_2x2(QC, Qc, e0, e1);
  const double c00 = QC[0], c01 = QC[1];
  const double c10 = QC[2], c11 = QC[3];
  const double r00 = Sq[0], r01 = Sq[1];
  const double r10 = Sq[2], r11 = Sq[3];
  /* tmp2 = QC * Sq */
  const double u00 = c00 * r00 + c01 * r10;
  const double u01 = c00 * r01 + c01 * r11;
  const double u10 = c10 * r00 + c11 * r10;
  const double u11 = c10 * r01 + c11 * r11;
  /* M = Sq * tmp2 */
  M_[0] = r00 * u00 + r01 * u10;
  M_[1] = r00 * u01 + r01 * u11;
  M_[2] = r10 * u00 + r11 * u10;
  M_[3] = r10 * u01 + r11 * u11;
}
