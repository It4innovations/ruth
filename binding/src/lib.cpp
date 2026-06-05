#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <thread>

#include "Routing/Algorithms/Alternatives/AlternativesPlateauAlgorithm.h"
#include "Routing/Data/Probability/ProfileStorageHDF5.h"
#include "Routing/Tests/IOUtils.h"

#include "comm/mpi_comm.hpp"
#include "scheduler/scheduler.hpp"
#include "workload/workload.hpp"
#include "workload/parallelfor.hpp"
#include "lib.hpp"

#define MULTIPLIER 16

void log(const std::string &message) {
  // Variable commented out to avoid unused variable warning
  // int rank = ace::CommInterface::instance()->get_rank();
  // std::cout << "--------------------------------\n"
  //           << "Rank " << rank << ": " << message
  //           << "\n--------------------------------" << std::endl<<std::flush;

  std::cout<< message << std::endl<<std::flush;
}

class State {
private:
  std::shared_ptr<Routing::Data::GraphMemory> graph_;
  std::shared_ptr<Routing::Algorithms::AlternativesPlateauAlgorithm> alg_;
  std::shared_ptr<std::vector<int>> vehicle_ids_;
  std::shared_ptr<std::vector<std::vector<std::vector<int>>>> routes_per_vehicle_;
  std::shared_ptr<std::vector<std::vector<float>>> travel_times_per_vehicle_;
  std::shared_ptr<std::vector<std::pair<int, float>>> route_travel_times_;
  std::mutex state_mutex;

  bool is_simulation_running_ = true;
  State() {}

public:
  static State &instance() {
    static State instance;
    return instance;
  }

  State(const State &) = delete;
  State &operator=(const State &) = delete;

  bool is_graph_set() { return graph_ != nullptr; }

  bool is_simulation_running() {
    std::lock_guard<std::mutex> guard(state_mutex);
    return is_simulation_running_;
  }

  void set_simulation_running(bool running) {
    std::lock_guard<std::mutex> guard(state_mutex);
    is_simulation_running_ = running;
  }

  void set_graph(const std::shared_ptr<Routing::Data::GraphMemory> graph) {
    if (graph_) {
      log("Graph already set");
    }
    graph_ = graph;

    auto settings = Routing::Algorithms::AlgorithmSettings();
    settings.filterSettings.allFilterOff = true;
    alg_ = std::make_shared<Routing::Algorithms::AlternativesPlateauAlgorithm>(graph, settings);
  }

  std::shared_ptr<Routing::Data::GraphMemory> get_graph() { return graph_; }

  std::shared_ptr<Routing::Algorithms::AlternativesPlateauAlgorithm> get_alg() {
    return alg_;
  }

  /// routes need to be protected with mutex as many threads may be willing to "push" new items. so i cannot return the shared pointer.
  // the get will return a shared ptr, but contextually reset the internal variable to another empty pointer.
  // the push will add one route to the vector
  // the init needs to be called only once, to setup the thing
  void init_routes() {
    if (vehicle_ids_) {
      log("already exists: i quit");
      throw std::runtime_error("routes_ ptr was already initialized");
    }
    std::lock_guard<std::mutex> guard(state_mutex);
    vehicle_ids_ = std::make_shared<std::vector<int>>();
    routes_per_vehicle_ = std::make_shared<std::vector<std::vector<std::vector<int>>>>();
    travel_times_per_vehicle_ = std::make_shared<std::vector<std::vector<float>>>();

    route_travel_times_ = std::make_shared<std::vector<std::pair<int, float>>>();
  }

  std::tuple<std::vector<int>, std::vector<std::vector<std::vector<int>>>,std::vector<std::vector<float>>>
      get_routes() {
    std::lock_guard<std::mutex> guard(state_mutex);

    std::vector<int> ret_ids = std::vector<int>();
    std::vector<std::vector<std::vector<int>>> ret_rpv = std::vector<std::vector<std::vector<int>>>();
    std::vector<std::vector<float>> ret_ttpv = std::vector<std::vector<float>>();

    std::swap(*vehicle_ids_, ret_ids);
    std::swap(*routes_per_vehicle_, ret_rpv);
    std::swap(*travel_times_per_vehicle_,ret_ttpv);

    // Sort by vehicle ID to maintain order
    if (!ret_ids.empty()) {
      const size_t n = ret_ids.size();
      std::vector<size_t> indices(n);
      std::iota(indices.begin(), indices.end(), 0);
      std::sort(indices.begin(), indices.end(),
                [&ret_ids](size_t i1, size_t i2) { return ret_ids[i1] < ret_ids[i2]; });

      std::vector<int> sorted_ids;
      std::vector<std::vector<std::vector<int>>> sorted_rpv;
      std::vector<std::vector<float>> sorted_ttpv;

      sorted_ids.reserve(n);
      sorted_rpv.reserve(n);
      sorted_ttpv.reserve(n);

      for (size_t i = 0; i < n; ++i) {
        sorted_ids.push_back(ret_ids[indices[i]]);
        sorted_rpv.push_back(std::move(ret_rpv[indices[i]]));
        sorted_ttpv.push_back(std::move(ret_ttpv[indices[i]]));
      }

      ret_ids = std::move(sorted_ids);
      ret_rpv = std::move(sorted_rpv);
      ret_ttpv = std::move(sorted_ttpv);
    }

    return std::make_tuple(ret_ids, ret_rpv, ret_ttpv);
  }

  std::vector<std::pair<int, float>> get_travel_times() {
    std::lock_guard<std::mutex> guard(state_mutex);
    std::vector<std::pair<int, float>> ret_travel_times;
    std::swap(*route_travel_times_, ret_travel_times);
    route_travel_times_ = std::make_shared<std::vector<std::pair<int, float>>>();
    return ret_travel_times;
  }

  void push_route(int v, std::vector<std::vector<int>> rpv, std::vector<float> ttpv) {
    std::lock_guard<std::mutex> guard(state_mutex);
    vehicle_ids_->emplace_back(v);
    routes_per_vehicle_->emplace_back(std::move(rpv));
    travel_times_per_vehicle_->emplace_back(std::move(ttpv));
  }

  void push_travel_time(int id, float travel_time) {
    std::lock_guard<std::mutex> guard(state_mutex);
    route_travel_times_->emplace_back(id, travel_time);
  }
};

/**
 * This workload updates the speeds of edges in the graph.
 *
 * It receives a vector of pairs, where each pair contains the edge id and the new speed.
 */
class UpdateSpeedsWorkload : public ace::workload {
public:
  std::vector<std::pair<int, float>> edge_speeds;

  UpdateSpeedsWorkload() : workload() {}

  UpdateSpeedsWorkload(const std::vector<std::pair<int, float>> &edge_speeds)
      : workload(), edge_speeds(edge_speeds) {}

  UpdateSpeedsWorkload(const std::shared_ptr<char> data, size_t size)
      : workload(data, size) {
    size_t offset = 0;
    size_t num_edges;
    std::memcpy(&num_edges, data.get() + offset, sizeof(size_t));
    offset += sizeof(size_t);

    edge_speeds.resize(num_edges);
    for (size_t i = 0; i < num_edges; ++i) {
      int edge_id;
      float speed;
      std::memcpy(&edge_id, data.get() + offset, sizeof(int));
      offset += sizeof(int);
      std::memcpy(&speed, data.get() + offset, sizeof(float));
      offset += sizeof(float);
      edge_speeds[i] = {edge_id, speed};
    }
  }

  void execute() override {
    auto graph = State::instance().get_graph();
    if (!graph)
    {
      log("No graph loaded, skipping workload");
      return;
    }

    for (const auto &item : edge_speeds)
    {
      int edgeId = item.first;
      float speed = item.second;
      graph->SetEdgeSpeed(edgeId, speed);
    }
  }

  size_t size() const override {
    return sizeof(size_t) + edge_speeds.size() * (sizeof(int) + sizeof(float));
  }

  char *marshal() const override {
    char *data = workload::marshal();
    size_t offset = 0;
    size_t num_edges = edge_speeds.size();

    std::memcpy(data + offset, &num_edges, sizeof(size_t));
    offset += sizeof(size_t);

    for (const auto &item : edge_speeds) {
      int edge_id = item.first;
      float speed = item.second;
      std::memcpy(data + offset, &edge_id, sizeof(int));
      offset += sizeof(int);
      std::memcpy(data + offset, &speed, sizeof(float));
      offset += sizeof(float);
    }
    return data;
  }
};

class LoadMapWorkload : public ace::workload {
public:
  std::string map_path = "";
  LoadMapWorkload() : workload() {}

  LoadMapWorkload(const std::string map_path)
      : workload(), map_path(map_path) {}

  LoadMapWorkload(const std::shared_ptr<char> data, size_t size)
      : workload(data, size) {
    size_t len;
    std::memcpy(&len, data.get(), sizeof(size_t));
    map_path = std::string(data.get() + sizeof(size_t), len);
  }

  void execute() override {
    // Variable commented out to avoid unused variable warning
    // int my_rank = ace::CommInterface::instance()->get_rank();

    if (State::instance().is_graph_set()) {
      log("No need to load graph again");
      return;
    }
    auto graph = Routing::Tests::LoadGraph(map_path);
    State::instance().set_graph(graph);
    log("Graph loaded");
  }

  size_t size() const override { return sizeof(size_t) + map_path.size(); }

  char *marshal() const override {
    char *data = workload::marshal();
    size_t len = map_path.size();
    // char *data = new char[sizeof(size_t) + len];
    std::memcpy(data, &len, sizeof(size_t));
    std::memcpy(data + sizeof(size_t), map_path.data(), len);
    return data;
  }
};

class CollectTravelTimesWorkload : public ace::workload {
public:
  std::vector<std::pair<int, float>> travel_times;
  CollectTravelTimesWorkload() : workload() {}
  CollectTravelTimesWorkload(const std::vector<std::pair<int, float>> &travel_times)
      : workload(), travel_times(travel_times) {}
  CollectTravelTimesWorkload(const std::shared_ptr<char> data, size_t size)
      : workload(data, size) {
    size_t offset = 0;
    size_t num_times;
    std::memcpy(&num_times, data.get() + offset, sizeof(size_t));
    offset += sizeof(size_t);

    travel_times.resize(num_times);
    for (size_t i = 0; i < num_times; ++i) {
      int id;
      float time;
      std::memcpy(&id, data.get() + offset, sizeof(int));
      offset += sizeof(int);
      std::memcpy(&time, data.get() + offset, sizeof(float));
      offset += sizeof(float);
      travel_times[i] = {id, time};
    }
  }

  void execute() override {
    for (const auto &tt : travel_times) {
      State::instance().push_travel_time(tt.first, tt.second);
    }
  }

  size_t size() const override {
    return sizeof(size_t) + travel_times.size() * (sizeof(int) + sizeof(float));
  }

  char *marshal() const override {
    char *data = workload::marshal();
    size_t offset = 0;
    size_t num_times = travel_times.size();

    std::memcpy(data + offset, &num_times, sizeof(size_t));
    offset += sizeof(size_t);

    for (const auto &tt : travel_times) {
      int id = tt.first;
      float time = tt.second;
      std::memcpy(data + offset, &id, sizeof(int));
      offset += sizeof(int);
      std::memcpy(data + offset, &time, sizeof(float));
      offset += sizeof(float);
    }
    return data;
  }
};

class CalculateTravelTimesWorkload : public ace::workload {
public:
  int start_index;
  std::vector<std::vector<int>> routes;
  const float INFINITE_TRAVEL_TIME = -1;

  CalculateTravelTimesWorkload(const int start_index, const std::vector<std::vector<int>> &routes)
      : workload(), start_index(start_index), routes(routes) {}

  CalculateTravelTimesWorkload(const std::shared_ptr<char> data, size_t size)
      : workload(data, size) {
    size_t offset = 0;
    std::memcpy(&start_index, data.get() + offset, sizeof(int));
    offset += sizeof(int);
    size_t num_routes;
    std::memcpy(&num_routes, data.get() + offset, sizeof(size_t));
    offset += sizeof(size_t);
    routes.resize(num_routes);
    for (size_t i = 0; i < num_routes; ++i) {
      size_t route_length;
      std::memcpy(&route_length, data.get() + offset, sizeof(size_t));
      offset += sizeof(size_t);

      routes[i].resize(route_length);
      std::memcpy(routes[i].data(), data.get() + offset, route_length * sizeof(int));
      offset += route_length * sizeof(int);
    }
  }

  void execute() override {

    const size_t chunk_size = std::max(1ul, routes.size() / (ace::Scheduler::instance()->num_workers() * MULTIPLIER));

    ace::parallelfor(
      ace::Range<size_t>(0, routes.size(), chunk_size),
      [](const size_t &i,
         const std::shared_ptr<std::vector<std::vector<int>>> routes_ptr,
         int start_index, float INFINITE_TRAVEL_TIME) {
        try {
          auto graph = State::instance().get_graph();
          if (!graph) {
            log("No graph loaded, skipping workload");
            return;
          }
          const auto &route = routes_ptr->at(i);
          const int request_index = start_index + i;

          float total_time = 0.0f;
          if (route.empty()) {
            log("Empty route for index " + std::to_string(i));
            State::instance().push_travel_time(request_index, INFINITE_TRAVEL_TIME);
            return;
          }

          auto node = graph->GetNodeById(route[0]);
          for (size_t j = 0; j + 1 < route.size(); ++j) {
            int next_id = route[j + 1];

            Routing::Edge *edge = nullptr;
            const auto &edges_out = node.GetEdgesOut();
            for (const auto &e : edges_out) {
              if (e->endNode.endNodePtr->id == next_id) {
                edge = e;
                break;
              }
            }

            if (!edge) {
              log("No edge found from " + std::to_string(node.id) + " to " + std::to_string(next_id));
              State::instance().push_travel_time(request_index, INFINITE_TRAVEL_TIME);
              return;
            }

            float speedMPS = static_cast<float>(edge->GetSpeed()) / 3.6f;
            if (speedMPS <= 0.0f) {
              State::instance().push_travel_time(request_index, INFINITE_TRAVEL_TIME);
              return;
            }
            float travelTime = edge->length / speedMPS;
            total_time += travelTime;

            node = *edge->endNode.endNodePtr;
          }

          State::instance().push_travel_time(request_index, total_time);
          return;
        }
        catch (const std::exception &e)
        {
          log("Error calculating travel time for route " + std::to_string(i) + ": " + e.what());
          State::instance().push_travel_time(start_index + i, INFINITE_TRAVEL_TIME);
        }
      },
    true,
    std::make_shared<std::vector<std::vector<int>>>(routes),
    start_index,
    INFINITE_TRAVEL_TIME);
  }

  size_t size() const override {
    size_t total = sizeof(int) + sizeof(size_t); // start_index + num_routes
    for (const auto &route : routes) {
      total += sizeof(size_t);             // route_length
      total += route.size() * sizeof(int); // route nodes
    }
    return total;
  }

  char *marshal() const override {
    char *data = workload::marshal();
    size_t offset = 0;

    std::memcpy(data + offset, &start_index, sizeof(int));
    offset += sizeof(int);

    size_t num_routes = routes.size();
    std::memcpy(data + offset, &num_routes, sizeof(size_t));
    offset += sizeof(size_t);

    for (const auto &route : routes) {
      size_t route_length = route.size();
      std::memcpy(data + offset, &route_length, sizeof(size_t));
      offset += sizeof(size_t);
      std::memcpy(data + offset, route.data(), route_length * sizeof(int));
      offset += route_length * sizeof(int);
    }

    return data;
  }
};

class CollectAlternativesWorkload : public ace::workload {
public:
  std::vector<int> vehicle_ids;
  std::vector<std::vector<std::vector<int>>> routes_per_vehicle;
  std::vector<std::vector<float>> travel_times_per_vehicle;

  ace::workload_types get_type() const override {return ace::workload_types::no_redist; }

  CollectAlternativesWorkload() : workload() {}

  CollectAlternativesWorkload(
      const std::vector<int> &vehicle_ids,
      const std::vector<std::vector<std::vector<int>>> &routes,
      const std::vector<std::vector<float>> &travel_times)
      : workload(), vehicle_ids(vehicle_ids), routes_per_vehicle(routes),
        travel_times_per_vehicle(travel_times) {}

  CollectAlternativesWorkload(const std::shared_ptr<char> data, size_t size)
      : workload(data, size) {
    size_t offset = 0;
    size_t num_vehicles;
    std::memcpy(&num_vehicles, data.get() + offset, sizeof(size_t));
    offset += sizeof(size_t);

    vehicle_ids.resize(num_vehicles);
    std::memcpy(vehicle_ids.data(), data.get() + offset,
                num_vehicles * sizeof(int));
    offset += num_vehicles * sizeof(int);

    routes_per_vehicle.resize(num_vehicles);
    travel_times_per_vehicle.resize(num_vehicles);

    for (size_t i = 0; i < num_vehicles; ++i) {
      size_t num_routes;
      std::memcpy(&num_routes, data.get() + offset, sizeof(size_t));
      offset += sizeof(size_t);

      routes_per_vehicle[i].resize(num_routes);
      travel_times_per_vehicle[i].resize(num_routes);

      for (size_t j = 0; j < num_routes; ++j) {
        size_t route_length;
        std::memcpy(&route_length, data.get() + offset, sizeof(size_t));
        offset += sizeof(size_t);

        routes_per_vehicle[i][j].resize(route_length);
        std::memcpy(routes_per_vehicle[i][j].data(), data.get() + offset,
                    route_length * sizeof(int));
        offset += route_length * sizeof(int);

        std::memcpy(&travel_times_per_vehicle[i][j], data.get() + offset,
                    sizeof(float));
        offset += sizeof(float);
      }
    }
  }

  void execute() override {
    for (size_t i = 0; i < vehicle_ids.size(); ++i) {
      State::instance().push_route(vehicle_ids[i], routes_per_vehicle[i], travel_times_per_vehicle[i]);
    }
  }

  size_t size() const override {
    size_t total = sizeof(size_t); // num_vehicles
    total += vehicle_ids.size() * sizeof(int);

    for (size_t i = 0; i < routes_per_vehicle.size(); ++i) {
      total += sizeof(size_t); // num_routes
      for (size_t j = 0; j < routes_per_vehicle[i].size(); ++j) {
        total += sizeof(size_t); // route_length
        total += routes_per_vehicle[i][j].size() * sizeof(int);
        total += sizeof(float); // travel time
      }
    }
    return total;
  }

  char *marshal() const override {
    char *data = workload::marshal();
    size_t offset = 0;
    size_t num_vehicles = vehicle_ids.size();

    std::memcpy(data + offset, &num_vehicles, sizeof(size_t));
    offset += sizeof(size_t);

    std::memcpy(data + offset, vehicle_ids.data(), num_vehicles * sizeof(int));
    offset += num_vehicles * sizeof(int);

    for (size_t i = 0; i < num_vehicles; ++i) {
      size_t num_routes = routes_per_vehicle[i].size();
      std::memcpy(data + offset, &num_routes, sizeof(size_t));
      offset += sizeof(size_t);

      for (size_t j = 0; j < num_routes; ++j) {
        size_t route_length = routes_per_vehicle[i][j].size();
        std::memcpy(data + offset, &route_length, sizeof(size_t));
        offset += sizeof(size_t);

        std::memcpy(data + offset, routes_per_vehicle[i][j].data(),
                    route_length * sizeof(int));
        offset += route_length * sizeof(int);

        std::memcpy(data + offset, &travel_times_per_vehicle[i][j],
                    sizeof(float));
        offset += sizeof(float);
      }
    }

    return data;
  }
};

class CollectWorkloads : public ace::workload {
public:
  CollectWorkloads() : workload() {}

  CollectWorkloads(const std::shared_ptr<char> data, size_t size)
      : workload(data, size) {
  }

  void execute() override {
    //get thing from status
    // auto retval = State::instance().get_routes();

    // if (std::get<0>(retval).size() > 0) {
    //   auto result = std::make_shared<CollectAlternativesWorkload>(
    //       std::get<0>(retval), std::get<1>(retval), std::get<2>(retval));
    //   ace::CommInterface::instance()->async_send(result, this->header.rank);
    //   result->wait();
    // }

    auto travel_times = State::instance().get_travel_times();
    if (travel_times.size() > 0) {
      auto travel_times_workload = std::make_shared<CollectTravelTimesWorkload>(travel_times);
      ace::CommInterface::instance()->async_send(travel_times_workload, this->header.rank);
      travel_times_workload->wait();
    }
  }

  size_t size() const override { return 0; }

  char *marshal() const override {
    char *data = workload::marshal();
    return data;
  }
  ace::workload_types get_type() const override {return ace::workload_types::no_redist; }
};

struct VehicleTask {
  int vehicle_id;
  int origin;
  int destination;
};

class AlternativesWorkload : public ace::workload {
public:
  std::vector<VehicleTask> vehicle_tasks;
  int max_routes;

  AlternativesWorkload() : workload(), max_routes(20) {}

  AlternativesWorkload(const std::vector<VehicleTask> &tasks, int max_routes)
      : workload(), vehicle_tasks(tasks), max_routes(max_routes) {}

  AlternativesWorkload(const std::shared_ptr<char> data, size_t size)
      : workload(data, size) {
    size_t offset = 0;

    size_t num_tasks;
    std::memcpy(&num_tasks, data.get() + offset, sizeof(size_t));
    offset += sizeof(size_t);

    vehicle_tasks.resize(num_tasks);
    std::memcpy(vehicle_tasks.data(), data.get() + offset,
                num_tasks * sizeof(VehicleTask));
    offset += num_tasks * sizeof(VehicleTask);

    std::memcpy(&max_routes, data.get() + offset, sizeof(int));
  }

  void execute() override {
    // std::vector<VehicleTask> ptr_to_data = std::vector<VehicleTask>(vehicle_tasks);
    auto alg = State::instance().get_alg();
    const size_t num_vehicles = vehicle_tasks.size();
    const size_t chunk_size = std::max(1ul, num_vehicles / (ace::Scheduler::instance()->num_workers() * MULTIPLIER));

    std::vector<int> vehicle_ids(num_vehicles);
    std::vector<std::vector<std::vector<int>>> routes_per_vehicle(num_vehicles);
    std::vector<std::vector<float>> travel_times_per_vehicle(num_vehicles);

    ace::parallelfor(
      ace::Range<size_t>(0, num_vehicles, chunk_size),
      [&](const size_t &i,
        const std::vector<VehicleTask>& ptr_to_data_in,
        const std::shared_ptr<Routing::Algorithms::AlternativesPlateauAlgorithm> &alg,
        int max_routes) {

        const VehicleTask& task = ptr_to_data_in[i];
        vehicle_ids[i] = task.vehicle_id;  // Always set vehicle ID to maintain order

        const std::unique_ptr<std::vector<Result>> routeResults = alg->GetResults(task.origin, task.destination, max_routes, false);

        if (routeResults && !routeResults->empty()) {
          std::vector<std::vector<int>> routes;
          std::vector<float> travel_times;
          for (const auto &result : *routeResults) {
            std::vector<int> segments;
            const auto &route = result.GetResult();
            if (!route.empty()) {
              segments.emplace_back(route[0].nodeId1);
              for (const auto &segment : route) {
                segments.emplace_back(segment.nodeId2);
              }
            }
            routes.emplace_back(segments);
            travel_times.emplace_back(result.travelTime);
          }

          std::swap(routes_per_vehicle[i],routes);
          std::swap(travel_times_per_vehicle[i],travel_times);
        }

        // If no routes found, vectors remain empty (default constructed)

      },
      true,
      vehicle_tasks,
      alg,
      max_routes);

      auto send_result = std::make_shared<CollectAlternativesWorkload>(vehicle_ids, routes_per_vehicle, travel_times_per_vehicle);
      ace::CommInterface::instance()->async_send(send_result, this->header.rank);
      send_result->wait();
  }

  size_t size() const override {
    return sizeof(size_t) + vehicle_tasks.size() * sizeof(VehicleTask) +
           sizeof(int);
  }

  char *marshal() const override {
    char *data = workload::marshal();
    size_t offset = 0;
    size_t num_tasks = vehicle_tasks.size();

    std::memcpy(data + offset, &num_tasks, sizeof(size_t));
    offset += sizeof(size_t);

    std::memcpy(data + offset, vehicle_tasks.data(),
                num_tasks * sizeof(VehicleTask));
    offset += num_tasks * sizeof(VehicleTask);

    std::memcpy(data + offset, &max_routes, sizeof(int));
    return data;
  }
};

class FinishSimulationWorkload : public ace::workload {
public:
  FinishSimulationWorkload() : workload() {}

  FinishSimulationWorkload(const std::shared_ptr<char> data, size_t size)
      : workload(data, size) {}

  void execute() override {
    State::instance().set_simulation_running(false);
  }

  size_t size() const override { return 0; }

  char *marshal() const override {
    char *data = workload::marshal();
    return data;
  }
};

namespace ruthlib{
  void setup_map(std::string path) {

    auto broadcast_task = std::make_shared<LoadMapWorkload>(path);
    auto comm = ace::CommInterface::instance();
    comm->async_broadcast(broadcast_task);

    broadcast_task->wait();

    std::cout << "Graph loaded on all nodes" << std::endl;
  }

  void setup_ace(int n_threads) {

    auto comm = ace::CommInterface::init<ace::MPIComm>(nullptr, nullptr);
    auto scheduler = ace::Scheduler::init<ace::Scheduler>(n_threads);

    comm->register_workload<LoadMapWorkload>();
    comm->register_workload<AlternativesWorkload>();
    comm->register_workload<CollectAlternativesWorkload>();
    comm->register_workload<CollectWorkloads>();
    comm->register_workload<CalculateTravelTimesWorkload>();
    comm->register_workload<CollectTravelTimesWorkload>();
    comm->register_workload<UpdateSpeedsWorkload>();
    comm->register_workload<FinishSimulationWorkload>();
  }

  void do_travel_times(std::vector<std::vector<int>> routes) {
    std::vector<std::shared_ptr<ace::workload>> distributed_tasks;

    auto comm = ace::CommInterface::instance();
    int num_ranks = comm->get_size();

    size_t size_offset = 0;
    size_t work_size = std::max(1ul, routes.size() / (num_ranks * MULTIPLIER));

    do {
      std::vector<std::vector<int>> work_list;
      size_t start_index = size_offset * work_size;
      for (size_t i = start_index; i < start_index + work_size && i < routes.size(); i++)
        work_list.push_back(routes[i]);

      if (work_list.empty()) break;

      auto task = std::make_shared<CalculateTravelTimesWorkload>(start_index, work_list);
      size_offset++;
      comm->async_send(task, size_offset % num_ranks);
      distributed_tasks.push_back(task);
    } while (size_offset * work_size < routes.size());

    for (auto &task : distributed_tasks) {
      task->wait();
    }
  }

  void do_alternatives(const int* od_data, size_t n_pairs, int max_routes) {
    std::vector<std::shared_ptr<ace::workload>> distributed_tasks;

    auto comm = ace::CommInterface::instance();
    int num_ranks = comm->get_size();

    size_t size_offset = 0;
    size_t work_size = std::max(1ul, n_pairs / (num_ranks * MULTIPLIER));

    do {
      std::vector<VehicleTask> work_list;

      for (size_t i = size_offset * work_size; i < (size_offset * work_size) + work_size && i < n_pairs; i++) {
        work_list.push_back({static_cast<int>(i), od_data[i * 2], od_data[i * 2 + 1]});
      }

      if (work_list.empty()) break;

      auto task = std::make_shared<AlternativesWorkload>(work_list, max_routes);
      size_offset++;
      comm->async_send(task, size_offset % num_ranks);
      distributed_tasks.push_back(task);

    } while (size_offset * work_size < n_pairs);

    for (auto &task : distributed_tasks) {
      task->wait();
    }

  }

  void do_alternatives(std::vector<std::pair<int, int>> OD_matrix, int max_routes) {
    std::vector<int> flat;
    flat.reserve(OD_matrix.size() * 2);
    for (const auto& p : OD_matrix) {
      flat.push_back(p.first);
      flat.push_back(p.second);
    }
    do_alternatives(flat.data(), OD_matrix.size(), max_routes);
  }

  bool is_master() {
    auto comm = ace::CommInterface::instance();
    return comm->master();
  }

  void barrier() {
    auto comm = ace::CommInterface::instance();
    comm->barrier();
  }

  void finalize() {
    auto comm = ace::CommInterface::instance();
    comm->finalize();
  }

  std::vector<std::pair<int, int>> load_od_matrix(const std::string &filepath,
                                                  int line_limit = 3) {
    std::vector<std::pair<int, int>> od_matrix;
    std::ifstream file(filepath);
    std::string line;

    if (!file.is_open()) {
      throw std::runtime_error("Could not open file: " + filepath);
    }

    // Skip header
    std::getline(file, line);

    int counter = 0;
    while (std::getline(file, line)) {
      if (counter >= line_limit) {
        break;
      }
      counter++;

      std::stringstream ss(line);
      std::string vehicle_id, node_from_str, node_to_str;

      if (!std::getline(ss, vehicle_id, ','))
        continue;
      if (!std::getline(ss, node_from_str, ','))
        continue;
      if (!std::getline(ss, node_to_str, ','))
        continue;

      int node_from = std::stoi(node_from_str);
      int node_to = std::stoi(node_to_str);
      od_matrix.emplace_back(node_from, node_to);
    }

    return od_matrix;
  }

  void init_routes() {
    State::instance().init_routes();
  }

  RoutesFlat get_routes_flat() {
    auto [vehicle_ids, routes_per_vehicle, travel_times_per_vehicle] = State::instance().get_routes();

    RoutesFlat flat;
    const size_t V = vehicle_ids.size();
    flat.vehicle_ids = std::move(vehicle_ids);
    flat.route_offsets.resize(V + 1, 0);

    // first pass: count totals so we can reserve before filling
    size_t total_routes = 0;
    size_t total_nodes = 0;
    for (size_t v = 0; v < V; ++v) {
      total_routes += routes_per_vehicle[v].size();
      for (const auto& route : routes_per_vehicle[v])
        total_nodes += route.size();
    }
    flat.travel_times.reserve(total_routes);
    flat.node_offsets.resize(total_routes + 1, 0);
    flat.nodes.reserve(total_nodes);

    // second pass: fill
    size_t route_idx = 0;
    for (size_t v = 0; v < V; ++v) {
      flat.route_offsets[v] = static_cast<int>(route_idx);
      for (size_t r = 0; r < routes_per_vehicle[v].size(); ++r) {
        flat.travel_times.push_back(travel_times_per_vehicle[v][r]);
        flat.node_offsets[route_idx] = static_cast<int>(flat.nodes.size());
        for (int node : routes_per_vehicle[v][r])
          flat.nodes.push_back(node);
        ++route_idx;
      }
    }
    flat.route_offsets[V] = static_cast<int>(route_idx);
    flat.node_offsets[total_routes] = static_cast<int>(flat.nodes.size());

    return flat;
  }

  std::vector<float> get_travel_times() {
    auto bc_wl = std::make_shared<CollectWorkloads>();
    ace::CommInterface::instance()->async_broadcast(bc_wl);
    bc_wl->wait();

    auto travel_times = State::instance().get_travel_times();

    // create vector of travel times sorted by route id
    std::vector<float> travel_times_vector;
    travel_times_vector.resize(travel_times.size());
    for (const auto &tt : travel_times)
    {
      if (tt.first >= 0 && static_cast<size_t>(tt.first) < travel_times_vector.size())
      {
        travel_times_vector[tt.first] = tt.second;
      }
      else
      {
        log("Invalid route ID " + std::to_string(tt.first) + ", skipping");
      }
    }

    return travel_times_vector;
  }

  void update_speeds(const int* edge_ids, const float* speeds, size_t n) {
    std::vector<std::pair<int, float>> edge_speeds(n);
    for (size_t i = 0; i < n; ++i)
      edge_speeds[i] = {edge_ids[i], speeds[i]};
    auto workload = std::make_shared<UpdateSpeedsWorkload>(edge_speeds);
    auto comm = ace::CommInterface::instance();
    comm->async_broadcast(workload);
    workload->wait();
  }

  void update_speeds(std::vector<std::pair<int, float>> edge_speeds) {
    std::vector<int> ids(edge_speeds.size());
    std::vector<float> spds(edge_speeds.size());
    for (size_t i = 0; i < edge_speeds.size(); ++i) {
      ids[i] = edge_speeds[i].first;
      spds[i] = edge_speeds[i].second;
    }
    update_speeds(ids.data(), spds.data(), edge_speeds.size());
  }

  bool is_simulation_running() {
    bool result = State::instance().is_simulation_running();
    return result;
  }

  void finish_simulation() {
    State::instance().set_simulation_running(false);
    auto comm = ace::CommInterface::instance();
    auto finish_workload = std::make_shared<FinishSimulationWorkload>();
    comm->async_broadcast(finish_workload);
    finish_workload->wait();
  }
} // namespace ruthlib
