// Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace carla {
namespace traffic_manager {

struct VehiclePIDConfig {
  std::vector<float> urban_longitudinal;   // [Kp, Ki, Kd]
  std::vector<float> highway_longitudinal; // [Kp, Ki, Kd]
  std::vector<float> urban_lateral;        // [Kp, Ki, Kd]
  std::vector<float> highway_lateral;      // [Kp, Ki, Kd]
};

// Default parameters
static const VehiclePIDConfig DEFAULT_PID_CONFIG = {
  {10.0f, 0.16f, 0.8f},
  {16.0f, 0.16f, 1.6f},
  {4.0f, 0.04f, 0.4f},
  {1.2f, 0.08f, 0.8f}
};

// Vehicle-specific PID parameters
static const std::unordered_map<std::string, VehiclePIDConfig> VEHICLE_PID_PARAMETERS = {
  
  {"vehicle.chevrolet.impala", {
    {10.0f, 0.16f, 0.8f},
    {16.0f, 0.16f, 1.6f},
    {4.0f, 0.02f, 0.08f},
    {0.8f, 0.04f, 0.8f}
  }}

};

inline const VehiclePIDConfig& GetVehiclePIDConfig(const std::string& vehicle_type_id) {
  auto it = VEHICLE_PID_PARAMETERS.find(vehicle_type_id);
  
  if (it != VEHICLE_PID_PARAMETERS.end()) {
    return it->second;
  }

  return DEFAULT_PID_CONFIG;
}

} // namespace traffic_manager
} // namespace carla