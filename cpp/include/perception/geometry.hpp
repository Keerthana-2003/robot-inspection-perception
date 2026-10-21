// Header-only C++17 perception kernels for edge deployment: IoU, class-aware NMS,
// LiDAR-to-camera projection and per-box LiDAR depth. Mirrors perception/boxes.py and
// perception/fusion.py so hot loops can move to C++ on the robot without changing behaviour.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <optional>
#include <vector>

namespace perception {

struct Box {
  float x1, y1, x2, y2;
  float area() const { return std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1); }
};

struct Intrinsics {
  double fx, fy, cx, cy;
  int width, height;
};

// Row-major 4x4 rigid transform (camera <- lidar).
using Transform = std::array<double, 16>;

struct Point3 {
  double x, y, z;
};

struct Projected {
  double u, v, depth;
  bool valid;
};

inline float iou(const Box& a, const Box& b) {
  const float ix = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float iy = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float inter = ix * iy;
  const float uni = a.area() + b.area() - inter;
  return uni > 0.f ? inter / uni : 0.f;
}

// Greedy NMS. Boxes of different classes never suppress each other. Returns kept indices.
inline std::vector<int> nms(const std::vector<Box>& boxes, const std::vector<float>& scores,
                            const std::vector<int>& classes, float iou_threshold) {
  std::vector<int> order(boxes.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int i, int j) { return scores[i] > scores[j]; });
  std::vector<bool> removed(boxes.size(), false);
  std::vector<int> keep;
  for (size_t a = 0; a < order.size(); ++a) {
    const int i = order[a];
    if (removed[i]) continue;
    keep.push_back(i);
    for (size_t b = a + 1; b < order.size(); ++b) {
      const int j = order[b];
      if (!removed[j] && classes[i] == classes[j] && iou(boxes[i], boxes[j]) > iou_threshold) removed[j] = true;
    }
  }
  return keep;
}

inline Projected project(const Point3& p_lidar, const Transform& T, const Intrinsics& K) {
  const double x = T[0] * p_lidar.x + T[1] * p_lidar.y + T[2] * p_lidar.z + T[3];
  const double y = T[4] * p_lidar.x + T[5] * p_lidar.y + T[6] * p_lidar.z + T[7];
  const double z = T[8] * p_lidar.x + T[9] * p_lidar.y + T[10] * p_lidar.z + T[11];
  if (z <= 1e-6) return {0, 0, z, false};
  const double u = K.fx * x / z + K.cx;
  const double v = K.fy * y / z + K.cy;
  const bool inside = u >= 0 && u < K.width && v >= 0 && v < K.height;
  return {u, v, z, inside};
}

// Robust depth of an object inside a 2D box: median of the nearest dense cluster of LiDAR
// returns (ignores background points that leak through the box). nullopt if too few points.
inline std::optional<double> box_depth(const Box& box, const std::vector<Point3>& points, const Transform& T,
                                       const Intrinsics& K, float shrink = 0.2f, size_t min_points = 3) {
  const float dx = (box.x2 - box.x1) * shrink / 2, dy = (box.y2 - box.y1) * shrink / 2;
  const Box inner{box.x1 + dx, box.y1 + dy, box.x2 - dx, box.y2 - dy};
  std::vector<double> depths;
  for (const auto& p : points) {
    const auto pr = project(p, T, K);
    if (pr.valid && pr.u >= inner.x1 && pr.u <= inner.x2 && pr.v >= inner.y1 && pr.v <= inner.y2)
      depths.push_back(pr.depth);
  }
  if (depths.size() < min_points) return std::nullopt;
  std::sort(depths.begin(), depths.end());
  auto median = [](const std::vector<double>& d) {
    const size_t n = d.size();
    return n % 2 ? d[n / 2] : 0.5 * (d[n / 2 - 1] + d[n / 2]);
  };
  if (depths.size() < 5) return median(depths);
  const double q1 = depths[depths.size() / 4];
  const double limit = q1 + std::max(0.5, 0.15 * q1);
  std::vector<double> near;
  for (double d : depths)
    if (d <= limit) near.push_back(d);
  return median(near);
}

}  // namespace perception
