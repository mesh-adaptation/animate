#include <math.h>
#include <Eigen/Dense>

using namespace Eigen;

#define PI 3.1415926535897932846

double distance(Vector2d p1, Vector2d p2) {
    return sqrt (pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
}

/*
  Gets the minimum angle at a vertex for all 
  triangular cell elements in a mesh.
*/
void get_min_angle(double *MinAngles, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector2d> p1((double *) &Coords[0]);
  Map<Vector2d> p2((double *) &Coords[2]);
  Map<Vector2d> p3((double *) &Coords[4]);

  // Compute edge vectors and distances
  Vector2d v12 = p2 - p1;
  Vector2d v23 = p3 - p2;
  Vector2d v13 = p3 - p1;
  double d12 = distance(p1, p2);
  double d23 = distance(p2, p3);
  double d13 = distance(p1, p3);

  // Compute angles from cosine formula
  double a1 = acos (v12.dot(v13) / (d12 * d13));
  double a2 = acos (-v12.dot(v23) / (d12 * d23));
  double a3 = acos (v23.dot(v13) / (d23 * d13));
  double aMin = std::min(a1, a2);
  MinAngles[0] = std::min(aMin, a3);
}

/*
  Gets the cell area for all triangular cell
  elements in a mesh.
*/
void get_area(double *Areas, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector2d> p1((double *) &Coords[0]);
  Map<Vector2d> p2((double *) &Coords[2]);
  Map<Vector2d> p3((double *) &Coords[4]);

  // Compute edge lengths
  double d12 = distance(p1, p2);
  double d23 = distance(p2, p3);
  double d13 = distance(p1, p3);
  double s = (d12 + d23 + d13) / 2;
  // Compute area using Heron's formula
  Areas[0] = sqrt(s * (s - d12) * (s - d23) * (s - d13));
}

/*
  Gets the equiangle skew for all triangular cell
  elements in a mesh.
*/
void get_eskew(double *ESkews, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector2d> p1((double *) &Coords[0]);
  Map<Vector2d> p2((double *) &Coords[2]);
  Map<Vector2d> p3((double *) &Coords[4]);

  // Compute edge vectors and distances
  Vector2d v12 = p2 - p1;
  Vector2d v23 = p3 - p2;
  Vector2d v13 = p3 - p1;
  double d12 = distance(p1, p2);
  double d23 = distance(p2, p3);
  double d13 = distance(p1, p3);

  // Compute angles from cosine formula
  double a1 = acos (v12.dot(v13) / (d12 * d13));
  double a2 = acos (-v12.dot(v23) / (d12 * d23));
  double a3 = acos (v23.dot(v13) / (d23 * d13));

  // Plug values into equiangle skew formula as per:
  // http://www.lcad.icmc.usp.br/~buscaglia/teaching/mfcpos2013/bakker_07-mesh.pdf
  double aMin = std::min(a1, a2);
  aMin = std::min(aMin, a3);
  double aMax = std::max(a1, a2);
  aMax = std::max(aMax, a3);
  double aIdeal = PI / 3;
  ESkews[0] = std::max((aMax - aIdeal / (PI - aIdeal)), (aIdeal - aMin) / aIdeal);
}

/*
  Gets the aspect ratio for all triangular cell
  elements in a mesh.
*/
void get_aspect_ratio(double *AspectRatios, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector2d> p1((double *) &Coords[0]);
  Map<Vector2d> p2((double *) &Coords[2]);
  Map<Vector2d> p3((double *) &Coords[4]);

  // Compute edge vectors and distances
  Vector2d v12 = p2 - p1;
  Vector2d v23 = p3 - p2;
  Vector2d v13 = p3 - p1;
  double d12 = distance(p1, p2);
  double d23 = distance(p2, p3);
  double d13 = distance(p1, p3);
  double s = (d12 + d23 + d13) / 2;

  // Calculate aspect ratio based on the circumradius and inradius as per:
  // https://stackoverflow.com/questions/10289752/aspect-ratio-of-a-triangle-of-a-meshed-surface
  AspectRatios[0] = (d12 * d23 * d13) / (8 * (s - d12) * (s - d23) * (s - d13));
}

/*
  Gets the scaled jacobian for all triangular cell
  elements in a mesh.
*/
void get_scaled_jacobian(double *SJacobians, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector2d> p1((double *) &Coords[0]);
  Map<Vector2d> p2((double *) &Coords[2]);
  Map<Vector2d> p3((double *) &Coords[4]);

  // Compute edge vectors and distances
  Vector2d v12 = p2 - p1;
  Vector2d v23 = p3 - p2;
  Vector2d v13 = p3 - p1;
  double d12 = distance(p1, p2);
  double d23 = distance(p2, p3);
  double d13 = distance(p1, p3);

  // Definition and calculation reference:
  // https://cubit.sandia.gov/15.5/help_manual/WebHelp/mesh_generation/mesh_quality_assessment/triangular_metrics.htm
  // https://www.osti.gov/biblio/5009
  double sj1 = std::abs(v12[0] * v13[1] - v13[0]*v12[1]) / (d12 * d13);
  double sj2 = std::abs(v12[0] * v23[1] - v23[0]*v12[1]) / (d12 * d23);
  double sj3 = std::abs(v23[0] * v13[1] - v13[0]*v23[1]) / (d13 * d23);
  SJacobians[0] = std::min(sj1, sj2);
  SJacobians[0] = std::min(sj3, SJacobians[0]);
}

/*
  Gets the skewness for all triangular cell
  elements in a mesh.
*/
void get_skewness(double *Skews, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector2d> p1((double *) &Coords[0]);
  Map<Vector2d> p2((double *) &Coords[2]);
  Map<Vector2d> p3((double *) &Coords[4]);

  // Calculating in accordance with:
  // https://www.engmorph.com/skewness-finite-elemnt
  Vector2d midPoint1 = p2 + (p3 - p2) / 2;
  Vector2d midPoint2 = p3 + (p1 - p3) / 2;
  Vector2d midPoint3 = p1 + (p2 - p1) / 2;

  Vector2d lineNormal1 = midPoint1 - p1;
  Vector2d lineOrth1 = midPoint3 - midPoint2;
  double t1 = acos (lineNormal1.dot(lineOrth1) / (distance(p1, midPoint1) * distance(midPoint2, midPoint3)));
  double t2 = PI - t1;
  double tMin = std::min(t1, t2);

  Vector2d lineNormal2 = midPoint2 - p2;
  Vector2d lineOrth2 = midPoint1 - midPoint3;
  double t3 = acos (lineNormal2.dot(lineOrth2) / (distance(p2, midPoint2) * distance(midPoint1, midPoint3)));
  double t4 = std::min(t3, PI - t3);
  tMin = std::min(tMin, t4);

  Vector2d lineNormal3 = midPoint3 - p3;
  Vector2d lineOrth3 = midPoint2 - midPoint1;
  double t5 = acos (lineNormal3.dot(lineOrth3) / (distance(p3, midPoint3) * distance(midPoint1, midPoint2)));
  double t6 = std::min(t3, PI - t5);
  tMin = std::min(tMin, t6);

  Skews[0] = PI/2 - tMin;
}

/*
  Gets the quality metric, based on a Riemannian metric
  for all triangular cell elements in a mesh.
*/
void get_metric(double *Metrics, double *Coords, const double *T_) {
    // Map coordinates onto Eigen objects
    Map<Vector2d> p1((double *) &Coords[0]);
    Map<Vector2d> p2((double *) &Coords[2]);
    Map<Vector2d> p3((double *) &Coords[4]);

    // Compute edge vectors and distances
    Vector2d v12 = p2 - p1;
    Vector2d v23 = p3 - p2;
    Vector2d v13 = p3 - p1;
    double d12 = distance(p1, p2);
    double d23 = distance(p2, p3);
    double d13 = distance(p1, p3);
    double s = (d12 + d23 + d13) / 2;
    double area = sqrt(s * (s-d12) * (s-d13) * (s-d23));

    // Map tensor  function as 2x2 Matrices
    Map<Matrix2d> M1((double *) &T_[0]);
    Map<Matrix2d> M2((double *) &T_[4]);
    Map<Matrix2d> M3((double *) &T_[8]);

    // Compute M(x, y) at centroid x_c to get area_M
    Matrix2d Mxc = (M1 + M2 + M3) / 3;
    double areaM = area * sqrt(Mxc.determinant());

    // Compute (squared) edge lengths in metric space
    double L1 = v23.dot(((M2 + M3)/2) * v23);
    double L2 = v13.dot(((M1 + M3)/2) * v13);
    double L3 = v12.dot(((M1 + M2)/2) * v12);

    // Calculated using Q_M formula in 2D, reference:
    // https://epubs.siam.org/doi/10.1137/090754078
    Metrics[0] = sqrt(3) * (L1 + L2 + L3) / (2 * areaM);
}
