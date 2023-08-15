#include <math.h>
#include <Eigen/Dense>

using namespace Eigen;

#define PI 3.1415926535897932846

double distance(Vector3d p1, Vector3d p2) {
    return sqrt (pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) + pow(p1[2] - p2[2], 2));
}

/*
  Gets the minimum (planar) angle for all
  tetrahedral cell elements in a mesh.
*/
void get_min_angle(double *MinAngles, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector3d> p1((double *) &Coords[0]);
  Map<Vector3d> p2((double *) &Coords[3]);
  Map<Vector3d> p3((double *) &Coords[6]);
  Map<Vector3d> p4((double *) &Coords[9]);

  // Compute edge vectors and distances
  Vector3d v12 = p2 - p1;
  Vector3d v13 = p3 - p1;
  Vector3d v14 = p4 - p1;
  Vector3d v23 = p3 - p2;
  Vector3d v24 = p4 - p2;
  Vector3d v34 = p4 - p3;

  double d12 = distance(p1, p2);
  double d13 = distance(p1, p3);
  double d14 = distance(p1, p4);
  double d23 = distance(p2, p3);
  double d24 = distance(p2, p4);
  double d34 = distance(p3, p4);

  double angles[12];
  // Compute angles from cosine formula
  angles[0] = acos(v13.dot(v14) / (d13 * d14));
  angles[1] = acos(v12.dot(v14) / (d12 * d14));
  angles[2] = acos(v13.dot(v12) / (d13 * d12));
  angles[3] = acos(v23.dot(v24) / (d23 * d24));
  angles[4] = acos(-v12.dot(v24) / (d12 * d24));
  angles[5] = acos(-v12.dot(v23) / (d12 * d23));
  angles[6] = acos(-v23.dot(v34) / (d23 * d34));
  angles[7] = acos(-v13.dot(v34) / (d13 * d34));
  angles[8] = acos(v13.dot(v23) / (d13 * d23));
  angles[9] = acos(v24.dot(v34) / (d24 * d34));
  angles[10] = acos(v14.dot(v34) / (d14 * d34));
  angles[11] = acos(v14.dot(v24) / (d14 * d24));

  double aMin = PI;
  for (int i = 0; i < 12; i++) {
    aMin = std::min(aMin, angles[i]);
  }

  MinAngles[0] = aMin;
}

/*
  Gets the volume for all tetrahedral cell
  elements in a mesh.
*/
void get_volume(double *Volumes, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector3d> p1((double *) &Coords[0]);
  Map<Vector3d> p2((double *) &Coords[3]);
  Map<Vector3d> p3((double *) &Coords[6]);
  Map<Vector3d> p4((double *) &Coords[9]);

  // Compute edge vectors
  Vector3d v12 = p2 - p1;
  Vector3d v13 = p3 - p1;
  Vector3d v14 = p4 - p1;
  Vector3d v23 = p3 - p2;
  Vector3d v24 = p4 - p2;
  Vector3d v34 = p4 - p3;

  Matrix3d volumeMatrix;
  for (int i = 0; i < 3; i++) {
    volumeMatrix(0, i) = v12[i];
    volumeMatrix(1, i) = v13[i];
    volumeMatrix(2, i) = v14[i];
  }
  Volumes[0] = std::abs(volumeMatrix.determinant() / 6);
}

/*
  Gets the equiangle skew for all tetrahedral cell
  elements in a mesh.
*/
void get_eskew(double *ESkews, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector3d> p1((double *) &Coords[0]);
  Map<Vector3d> p2((double *) &Coords[3]);
  Map<Vector3d> p3((double *) &Coords[6]);
  Map<Vector3d> p4((double *) &Coords[9]);

  // Compute edge vectors and distances
  Vector3d v12 = p2 - p1;
  Vector3d v13 = p3 - p1;
  Vector3d v14 = p4 - p1;
  Vector3d v23 = p3 - p2;
  Vector3d v24 = p4 - p2;
  Vector3d v34 = p4 - p3;

  double d12 = distance(p1, p2);
  double d13 = distance(p1, p3);
  double d14 = distance(p1, p4);
  double d23 = distance(p2, p3);
  double d24 = distance(p2, p4);
  double d34 = distance(p3, p4);

  double angles[12];
  // Compute angles from cosine formula
  angles[0] = acos(v13.dot(v14) / (d13 * d14));
  angles[1] = acos(v12.dot(v14) / (d12 * d14));
  angles[2] = acos(v13.dot(v12) / (d13 * d12));
  angles[3] = acos(v23.dot(v24) / (d23 * d24));
  angles[4] = acos(-v12.dot(v24) / (d12 * d24));
  angles[5] = acos(-v12.dot(v23) / (d12 * d23));
  angles[6] = acos(-v23.dot(v34) / (d23 * d34));
  angles[7] = acos(-v13.dot(v34) / (d13 * d34));
  angles[8] = acos(-v13.dot(-v23) / (d13 * d23));
  angles[9] = acos(-v24.dot(-v34) / (d24 * d34));
  angles[10] = acos(-v14.dot(-v34) / (d14 * d34));
  angles[11] = acos(-v14.dot(-v24) / (d14 * d24));

  double aMin = PI;
  double aMax = 0.0;
  for (int i = 0; i < 12; i++) {
    aMin = std::min(aMin, angles[i]);
    aMax = std::max(aMax, angles[i]);
  }
  double aIdeal = PI / 3;
  ESkews[0] = std::max((aMax - aIdeal) / (PI - aIdeal), (aIdeal - aMin) / aIdeal);
}

/*
  Gets the aspect raio for all tetrahedral cell
  elements in a mesh.
*/
void get_aspect_ratio(double *AspectRatios, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector3d> p1((double *) &Coords[0]);
  Map<Vector3d> p2((double *) &Coords[3]);
  Map<Vector3d> p3((double *) &Coords[6]);
  Map<Vector3d> p4((double *) &Coords[9]);

  // Compute edge vectors and distances
  Vector3d v12 = p2 - p1;
  Vector3d v13 = p3 - p1;
  Vector3d v14 = p4 - p1;
  Vector3d v23 = p3 - p2;
  Vector3d v24 = p4 - p2;
  Vector3d v34 = p4 - p3;

  double d12 = distance(p1, p2);
  double d13 = distance(p1, p3);
  double d14 = distance(p1, p4);
  double d23 = distance(p2, p3);
  double d24 = distance(p2, p4);
  double d34 = distance(p3, p4);

  Matrix3d volumeMatrix;
  for (int i = 0; i < 3; i++) {
    volumeMatrix(0, i) = v12[i];
    volumeMatrix(1, i) = v13[i];
    volumeMatrix(2, i) = v14[i];
  }
  double volume = std::abs(volumeMatrix.determinant() / 6);

  // Reference for inradius and circumradius calculations on the tetrahedron
  // https://en.wikipedia.org/wiki/Tetrahedron#Inradius
  double cir_radius = sqrt((d12 * d34 + d13 * d24 + d14 * d23) *
                           (d12 * d34 + d13 * d24 - d14 * d23) *
                           (d12 * d34 - d13 * d24 + d14 * d23) *
                           (-d12 * d34 + d13 * d24 + d14 * d23)) / (24 * volume);

  double s1 = (d23 + d24 + d34) / 2;
  double s2 = (d13 + d14 + d34) / 2;
  double s3 = (d12 + d14 + d24) / 2;
  double s4 = (d12 + d13 + d23) / 2;
  double f_area1 = sqrt(s1 * (s1 - d23) * (s1 - d24) * (s1 - d34));
  double f_area2 = sqrt(s2 * (s2 - d13) * (s2 - d14) * (s2 - d34));
  double f_area3 = sqrt(s3 * (s3 - d12) * (s3 - d14) * (s3 - d24));
  double f_area4 = sqrt(s4 * (s4 - d12) * (s4 - d13) * (s4 - d23));
  double in_radius = 3 * volume / (f_area1 + f_area2 + f_area3 + f_area4);

  AspectRatios[0] = cir_radius / (3 * in_radius);
}

/*
  Gets the scaled jacobian for all tetrahedral cell
  elements in a mesh.
*/
void get_scaled_jacobian(double *SJacobians, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector3d> p1((double *) &Coords[0]);
  Map<Vector3d> p2((double *) &Coords[3]);
  Map<Vector3d> p3((double *) &Coords[6]);
  Map<Vector3d> p4((double *) &Coords[9]);

  // Compute edge vectors and distances
  Vector3d v12 = p2 - p1;
  Vector3d v13 = p3 - p1;
  Vector3d v14 = p4 - p1;
  Vector3d v23 = p3 - p2;
  Vector3d v24 = p4 - p2;
  Vector3d v34 = p4 - p3;

  double d12 = distance(p1, p2);
  double d13 = distance(p1, p3);
  double d14 = distance(p1, p4);
  double d23 = distance(p2, p3);
  double d24 = distance(p2, p4);
  double d34 = distance(p3, p4);

  Matrix3d M1, M2, M3, M4;
  double sj[4];
  for (int i = 0; i < 3; i++) {
    M1(0, i) = v12[i];
    M1(1, i) = v13[i];
    M1(2, i) = v14[i];

    M2(0, i) = -v12[i];
    M2(1, i) = v23[i];
    M2(2, i) = v24[i];

    M3(0, i) = -v13[i];
    M3(1, i) = -v23[i];
    M3(2, i) = v34[i];

    M4(0, i) = -v14[i];
    M4(1, i) = -v24[i];
    M4(2, i) = -v34[i];
  }
  sj[0] = std::abs(M1.determinant()) / (d12 * d13 * d14);
  sj[1] = std::abs(M2.determinant()) / (d12 * d23 * d24);
  sj[2] = std::abs(M3.determinant()) / (d13 * d23 * d34);
  sj[3] = std::abs(M4.determinant()) / (d14 * d24 * d34);

  SJacobians[0] = std::min(sj[0], sj[1]);
  SJacobians[0] = std::min(SJacobians[0], sj[2]);
  SJacobians[0] = std::min(SJacobians[0], sj[3]);
}

/*
  Gets the quality metric, based on a Riemannian metric
  for all tetrahedral cell elements in a mesh.
*/
void get_metric(double *Metrics, double *Coords, const double *T_) {
  // Map vertices as vectors
  Map<Vector3d> p1((double *) &Coords[0]);
  Map<Vector3d> p2((double *) &Coords[3]);
  Map<Vector3d> p3((double *) &Coords[6]);
  Map<Vector3d> p4((double *) &Coords[9]);

  // Precompute some vectors, and distances
  Vector3d v12 = p2 - p1;
  Vector3d v13 = p3 - p1;
  Vector3d v14 = p4 - p1;
  Vector3d v23 = p3 - p2;
  Vector3d v24 = p4 - p2;
  Vector3d v34 = p4 - p3;

  double d12 = distance(p1, p2);
  double d13 = distance(p1, p3);
  double d14 = distance(p1, p4);
  double d23 = distance(p2, p3);
  double d24 = distance(p2, p4);
  double d34 = distance(p3, p4);

  Matrix3d volMatrix;
  for (int i = 0; i < 3; i++) {
    volMatrix(0, i) = v12[i];
    volMatrix(1, i) = v13[i];
    volMatrix(2, i) = v14[i];
  }

  double volume = std::abs(volMatrix.determinant()) / 6;

  // Map tensor as 3x3 Matrices
  Map<Matrix3d> M1((double *) &T_[0]);
  Map<Matrix3d> M2((double *) &T_[9]);
  Map<Matrix3d> M3((double *) &T_[18]);
  Map<Matrix3d> M4((double *) &T_[27]);

  // Compute M(x, y) at centroid x_c to get area_M
  Matrix3d Mxc = (M1 + M2 + M3 + M4) / 3;
  double volumeM = volume * sqrt(Mxc.determinant());

  // Compute (squared) edge lengths in metric
  double L1 = v12.dot(((M1 + M2)/2) * v12);
  double L2 = v13.dot(((M1 + M3)/2) * v13);
  double L3 = v14.dot(((M1 + M4)/2) * v14);
  double L4 = v23.dot(((M2 + M3)/2) * v23);
  double L5 = v24.dot(((M2 + M4)/2) * v24);
  double L6 = v34.dot(((M3 + M4)/2) * v34);

  Metrics[0] = sqrt(3) * (L1 + L2 + L3 + L4 + L5 + L6) / (216 * volumeM);
}
