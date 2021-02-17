// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <memory>
#include <thread>
#include <future>
#include <condition_variable>
#include "core/framework/ml_value.h"

namespace onnxruntime {
namespace contrib {

class OrtTasks final {
 public:
  static OrtTasks& GetInstance() {
    static OrtTasks instance_;
    return instance_;
  }

  void PrepareForegroundWait();
  void WaitInForegroundThread();
  void WakeupForegroundThread();

  void CreateBackgroundTask(bool* terminate_flags);
  void PrepareBackgroundWait();
  void WaitInBackgroundThread();
  void WakeupBackgroundThread(int64_t run_id);

  void SetForwardOutputs(const std::vector<OrtValue>& forward_outputs);
  std::vector<OrtValue> GetForwardOutputs(int64_t run_id);

  void SetBackwardInputs(int64_t run_id, const std::vector<OrtValue>& backward_inputs);
  std::vector<OrtValue> GetBackwardInputs();

  void SetStatus(const Status& status);
  bool StatusIsReady(int64_t run_id);
  bool StatusIsValid(int64_t run_id);
  Status GetStatus(int64_t run_id);

  void SetTerminateFlag(int64_t run_id) {
    *(bg_tasks[run_id]->terminate_flags_) = true;
  }

 private:
  OrtTasks() = default;
  ~OrtTasks() = default;
  OrtTasks(const OrtTasks&) = delete;
  OrtTasks& operator=(const OrtTasks&) = delete;

  struct Task {
    std::atomic<bool> signaled;
    mutable std::mutex mutex;
    mutable std::condition_variable cv;

    std::promise<std::vector<OrtValue>> forward_output_promise_;
    std::future<std::vector<OrtValue>> forward_output_future_ = forward_output_promise_.get_future();

    std::promise<std::vector<OrtValue>> backward_input_promise_;
    std::future<std::vector<OrtValue>> backward_input_future_ = backward_input_promise_.get_future();

    std::promise<Status> status_promise_;
    std::future<Status> status_future_ = status_promise_.get_future();

    bool* terminate_flags_;

    Task() {
      signaled.store(false);
    }
  };

  std::hash<std::thread::id> hasher_;
  Task fg_task;
  std::unordered_map<int64_t, std::unique_ptr<Task>> bg_tasks;
};

}  // namespace contrib
}  // namespace onnxruntime
