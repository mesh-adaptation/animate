/*
 * Plain-C ports of the Eigen-based metric kernels formerly in metric3d.cxx.
 *
 * All matrices are stored row-major; metrics are 3x3 symmetric positive
 * definite. Eigenvalue problems are delegated to LAPACK dsyev via PETSc's
 * <petscblaslapack.h> (which gives us the PetscBLASInt / PetscReal typedefs
 * and the LAPACKsyev_ name-mangled symbol).
 *
 * See metric2d.c for the PREAMBLE-BEGIN/END section convention used by
 * animate._kernel.get_metric_kernel.
 */

/* PREAMBLE-BEGIN */

#include <math.h>
#include <petscblaslapack.h>

/*
 * Solve the eigenvalue problem for a symmetric 3x3 row-major matrix,
 * storing eigenvectors as columns of Q and eigenvalues in D in *ascending*
 * order (LAPACK's natural ordering).
 *
 * LAPACK expects column-major input, but a real symmetric matrix satisfies
 * A = A^T, so passing a row-major buffer is equivalent to passing the
 * transpose in column-major — which is the same matrix. The returned
 * eigenvector buffer is then also row-major with eigenvectors as columns.
 */
static void eig3x3(double Q[9], double D[3], const double A[9]) {
  char jobz = 'V';
  char uplo = 'U';
  PetscBLASInt n = 3;
  PetscBLASInt lda = 3;
  PetscBLASInt lwork = 34;  /* dsyev has lwork >= 3*n - 1; allow headroom. */
  PetscBLASInt info = 0;
  double work[34];

  /* Copy A into Q (dsyev overwrites the input with eigenvectors). */
  for (int i = 0; i < 9; i++) Q[i] = A[i];
  LAPACKsyev_(&jobz, &uplo, &n, Q, &lda, D, work, &lwork, &info);
  /* info != 0 should not occur for a 3x3; if it does, behaviour will be
   * clearly wrong downstream, which is preferable to silently corrupting. */
  (void)info;

  /* LAPACK works on column-major data; feeding our row-major symmetric A
   * means it sees A^T = A (same matrix). But the eigenvector matrix it
   * writes is stored column-major from our perspective, i.e. its columns
   * are rows of our Q buffer. Transpose Q in-place to restore row-major
   * column-vector storage.
   */
  for (int i = 0; i < 3; i++)
    for (int j = i + 1; j < 3; j++) {
      const double t = Q[3 * i + j];
      Q[3 * i + j] = Q[3 * j + i];
      Q[3 * j + i] = t;
    }
}

/*
 * Compute the matrix product A = Q diag(d0, d1, d2) Q^T for a 3x3 row-major
 * orthogonal matrix Q.
 */
static void rqdt_3x3(double M[9], const double Q[9], double d0, double d1, double d2) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      M[3 * i + j] = d0 * Q[3 * i + 0] * Q[3 * j + 0]
                   + d1 * Q[3 * i + 1] * Q[3 * j + 1]
                   + d2 * Q[3 * i + 2] * Q[3 * j + 2];
    }
  }
}

/* PREAMBLE-END */

/*
 * Extract eigenvectors and eigenvalues from a 3D metric field.
 */
void get_eigendecomposition(double EVecs_[9], double EVals_[3], const double * M_) {
  eig3x3(EVecs_, EVals_, M_);
}

/*
 * Extract eigenvectors and eigenvalues from a 3D metric field, with
 * eigenvalues **decreasing** in magnitude.
 */
void get_reordered_eigendecomposition(double EVecs_[9], double EVals_[3], const double * M_) {
  double D[3];
  double Q[9];
  eig3x3(Q, D, M_);

  /* Sort columns of Q and entries of D by descending |D|. */
  int idx[3] = {0, 1, 2};
  const double a0 = fabs(D[0]);
  const double a1 = fabs(D[1]);
  const double a2 = fabs(D[2]);
  /* Simple 3-element insertion sort on idx by descending |D[idx[]]|. */
  if (a1 > a0) { idx[0] = 1; idx[1] = 0; } else { idx[0] = 0; idx[1] = 1; }
  idx[2] = 2;
  if (fabs(D[idx[2]]) > fabs(D[idx[1]])) {
    int tmp = idx[1]; idx[1] = idx[2]; idx[2] = tmp;
    if (fabs(D[idx[1]]) > fabs(D[idx[0]])) {
      tmp = idx[0]; idx[0] = idx[1]; idx[1] = tmp;
    }
  }
  for (int k = 0; k < 3; k++) {
    EVals_[k] = D[idx[k]];
    for (int i = 0; i < 3; i++) {
      EVecs_[3 * i + k] = Q[3 * i + idx[k]];
    }
  }
}

/*
 * Post-process a 3D metric field in order to enforce maximum and minimum
 * tolerated metric magnitudes, as well as maximum tolerated anisotropy.
 */
void postproc_metric(double A_[9], const double * h_min, const double * h_max, const double * a_max) {
  const double l_max = pow(*h_min, -2);
  const double l_min = pow(*h_max, -2);
  const double la_min = pow(*a_max, -2);

  double Q[9];
  double D[3];
  eig3x3(Q, D, A_);

  /* Enforce maximum and minimum tolerated metric magnitudes. */
  double d0 = fmin(l_max, fmax(l_min, fabs(D[0])));
  double d1 = fmin(l_max, fmax(l_min, fabs(D[1])));
  double d2 = fmin(l_max, fmax(l_min, fabs(D[2])));
  const double max_eig = fmax(fmax(d0, d1), d2);

  /* Enforce maximum tolerated anisotropy. */
  d0 = fmax(d0, la_min * max_eig);
  d1 = fmax(d1, la_min * max_eig);
  d2 = fmax(d2, la_min * max_eig);

  /* Reassemble the metric with the clamped eigenvalues. */
  rqdt_3x3(A_, Q, d0, d1, d2);
}

/*
 * Intersect two 3D metric fields.
 */
void intersect(double M_[9], const double * A_, const double * B_) {
  /* Eigendecomposition of A: A = Qa Da Qa^T. */
  double Qa[9];
  double Da[3];
  eig3x3(Qa, Da, A_);

  /* Square root and inverse square root of A. */
  double Sq[9];
  double Sqi[9];
  rqdt_3x3(Sq, Qa, sqrt(Da[0]), sqrt(Da[1]), sqrt(Da[2]));
  rqdt_3x3(Sqi, Qa, 1.0 / sqrt(Da[0]), 1.0 / sqrt(Da[1]), 1.0 / sqrt(Da[2]));

  /* C = Sqi^T B Sqi (Sqi symmetric). */
  double tmp[9];
  double C[9];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      tmp[3 * i + j] = 0.0;
      for (int k = 0; k < 3; k++)
        tmp[3 * i + j] += B_[3 * i + k] * Sqi[3 * k + j];
    }
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      C[3 * i + j] = 0.0;
      for (int k = 0; k < 3; k++)
        C[3 * i + j] += Sqi[3 * i + k] * tmp[3 * k + j];  /* Sqi^T = Sqi */
    }

  /* Eigendecomposition of C = Qc Dc Qc^T; clamp eigenvalues below at 1. */
  double Qc[9];
  double Dc[3];
  eig3x3(Qc, Dc, C);
  const double e0 = fmax(Dc[0], 1.0);
  const double e1 = fmax(Dc[1], 1.0);
  const double e2 = fmax(Dc[2], 1.0);

  /* M = Sq^T (Qc diag(e) Qc^T) Sq; both factors symmetric. */
  double QC[9];
  rqdt_3x3(QC, Qc, e0, e1, e2);
  double tmp2[9];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      tmp2[3 * i + j] = 0.0;
      for (int k = 0; k < 3; k++)
        tmp2[3 * i + j] += QC[3 * i + k] * Sq[3 * k + j];
    }
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      M_[3 * i + j] = 0.0;
      for (int k = 0; k < 3; k++)
        M_[3 * i + j] += Sq[3 * i + k] * tmp2[3 * k + j];  /* Sq^T = Sq */
    }
}
