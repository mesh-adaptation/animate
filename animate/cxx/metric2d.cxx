#include <Eigen/Dense>

using namespace Eigen;

void get_eigendecomposition_private(double EVecs_[4], double EVals_[2], const double * M_, bool reorder) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, 2, 2, RowMajor> > EVecs((double *)EVecs_);
  Map<Vector2d> EVals((double *)EVals_);
  Map<Matrix<double, 2, 2, RowMajor> > M((double *)M_);

  // Solve eigenvalue problem
  SelfAdjointEigenSolver<Matrix<double, 2, 2, RowMajor>> eigensolver(M);
  Matrix<double, 2, 2, RowMajor> Q = eigensolver.eigenvectors();
  Vector2d D = eigensolver.eigenvalues();

  // Reorder eigenpairs by magnitude of eigenvalue
  if (reorder) {
    if (fabs(D(0)) > fabs(D(1))) {
      EVecs = Q;
      EVals = D;
    } else {
      EVecs(0,0) = Q(0,1);EVecs(0,1) = Q(0,0);
      EVecs(1,0) = Q(1,1);EVecs(1,1) = Q(1,0);
      EVals(0) = D(1);
      EVals(1) = D(0);
    }
  } else {
    EVecs = Q;
    EVals = D;
  }
}

/*
  Extract eigenvectors and eigenvalues from
  a 2D metric field.
*/
void get_eigendecomposition(double EVecs_[4], double EVals_[2], const double * M_) {
  get_eigendecomposition_private(EVecs_, EVals_, M_, false);
}

/*
  Extract eigenvectors and eigenvalues from
  a 2D metric field, with eigenvalues
  **decreasing** in magnitude.
*/
void get_reordered_eigendecomposition(double EVecs_[4], double EVals_[2], const double * M_) {
  get_eigendecomposition_private(EVecs_, EVals_, M_, true);
}

/*
  Construct a 2D metric from eigenvectors and
  eigenvalues as an orthogonal eigendecomposition.
*/
void set_eigendecomposition(double M_[4], const double * EVecs_, const double * EVals_) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, 2, 2, RowMajor> > M((double *)M_);
  Map<Matrix<double, 2, 2, RowMajor> > EVecs((double *)EVecs_);
  Map<Vector2d> EVals((double *)EVals_);

  // Compute metric from eigendecomposition
  M = EVecs * EVals.asDiagonal() * EVecs.transpose();
}

/*
  Modify the eigenvalues of a 2D Hessian matrix so
  that it is positive-definite.
*/
void metric_from_hessian(double A_[4], const double * B_) {
  double l_min = 1.0e-60;
  double l_max = 1.0e+60;

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, 2, 2, RowMajor> > A((double *)A_);
  Map<Matrix<double, 2, 2, RowMajor> > B((double *)B_);

  // Compute mean diagonal and set values appropriately
  B(0,1) = 0.5*(B(0,1) + B(1,0));
  B(1,0) = B(0,1);

  // Solve eigenvalue problem
  SelfAdjointEigenSolver<Matrix<double, 2, 2, RowMajor>> eigensolver(B);
  Matrix<double, 2, 2, RowMajor> Q = eigensolver.eigenvectors();
  Vector2d D = eigensolver.eigenvalues();

  // Take modulus of eigenvalues
  D(0) = fmin(l_max, fmax(l_min, abs(D(0))));
  D(1) = fmin(l_max, fmax(l_min, abs(D(1))));

  // Build metric from eigendecomposition
  A += Q * D.asDiagonal() * Q.transpose();
}

/*
  Post-process a 2D metric field in order to enforce
  maximum and minimum tolerated metric magnitudes,
  as well as maximum tolerated anisotropy.
*/
void postproc_metric(double A_[4], const double * h_min, const double * h_max, const double * a_max)
{
  double l_max = pow(*h_min, -2);
  double l_min = pow(*h_max, -2);
  double la_min = pow(*a_max, -2);

  // Map input/output metric onto an Eigen object
  Map<Matrix<double, 2, 2, RowMajor> > A((double *)A_);

  // Solve eigenvalue problem
  SelfAdjointEigenSolver<Matrix<double, 2, 2, RowMajor>> eigensolver(A);
  Matrix<double, 2, 2, RowMajor> Q = eigensolver.eigenvectors();
  Vector2d D = eigensolver.eigenvalues();

  // Enforce maximum and minimum tolerated metric magnitudes
  D(0) = fmin(l_max, fmax(l_min, abs(D(0))));
  D(1) = fmin(l_max, fmax(l_min, abs(D(1))));
  double max_eig = fmax(D(0), D(1));

  // Enforce maximum tolerated anisotropy
  D(0) = fmax(D(0), la_min * max_eig);
  D(1) = fmax(D(1), la_min * max_eig);

  // Build metric from eigendecomposition
  A = Q * D.asDiagonal() * Q.transpose();
}

/*
  Intersect two 2D metric fields.
*/
void intersect(double M_[4], const double * A_, const double * B_) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, 2, 2, RowMajor> > M((double *)M_);
  Map<Matrix<double, 2, 2, RowMajor> > A((double *)A_);
  Map<Matrix<double, 2, 2, RowMajor> > B((double *)B_);

  // Solve eigenvalue problem of first metric, taking square root of eigenvalues
  SelfAdjointEigenSolver<Matrix<double, 2, 2, RowMajor>> eigensolver(A);
  Matrix<double, 2, 2, RowMajor> Q = eigensolver.eigenvectors();
  Matrix<double, 2, 2, RowMajor> D = eigensolver.eigenvalues().array().sqrt().matrix().asDiagonal();

  // Compute square root and inverse square root metrics
  Matrix<double, 2, 2, RowMajor> Sq = Q * D * Q.transpose();
  Matrix<double, 2, 2, RowMajor> Sqi = Q * D.inverse() * Q.transpose();

  // Solve eigenvalue problem for triple product of inverse square root metric and the second metric
  SelfAdjointEigenSolver<Matrix<double, 2, 2, RowMajor>> eigensolver2(Sqi.transpose() * B * Sqi);
  Q = eigensolver2.eigenvectors();
  D = eigensolver2.eigenvalues().array().max(1).matrix().asDiagonal();

  // Compute metric intersection
  M = Sq.transpose() * Q * D * Q.transpose() * Sq;
}
