#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

#include "perception/geometry.hpp"

using namespace perception;

static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

int main() {
  // IoU
  assert(near(iou({0, 0, 10, 10}, {1, 1, 11, 11}), 81.0 / 119.0, 1e-6));
  assert(iou({0, 0, 1, 1}, {5, 5, 6, 6}) == 0.f);

  // class-aware NMS
  std::vector<Box> boxes{{0, 0, 10, 10}, {1, 1, 11, 11}, {0, 0, 10, 10}, {50, 50, 60, 60}};
  auto keep = nms(boxes, {0.9f, 0.8f, 0.7f, 0.6f}, {0, 0, 1, 0}, 0.5f);
  assert((keep == std::vector<int>{0, 2, 3}));

  // projection: lidar (x fwd, y left, z up) -> camera (x right, y down, z fwd)
  const Intrinsics K{800, 800, 640, 360, 1280, 720};
  const Transform T{0, -1, 0, 0,  0, 0, -1, 0,  1, 0, 0, 0,  0, 0, 0, 1};
  auto p = project({5, 0, 0}, T, K);
  assert(p.valid && near(p.u, 640, 1e-9) && near(p.v, 360, 1e-9) && near(p.depth, 5, 1e-9));
  assert(!project({-5, 0, 0}, T, K).valid);  // behind the camera

  // box depth ignores the background wall
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> U(-1, 1);
  std::vector<Point3> pts;
  for (int i = 0; i < 400; ++i) pts.push_back({5.0, 0.25 * U(rng), 0.5 * U(rng)});
  for (int i = 0; i < 3000; ++i) pts.push_back({20.0, 8 * U(rng), 3 * U(rng)});
  auto d = box_depth({600, 280, 680, 440}, pts, T, K);
  assert(d && near(*d, 5.0, 1e-6));
  assert(!box_depth({0, 0, 10, 10}, {}, T, K));

  std::puts("all C++ geometry tests passed");
  return 0;
}
