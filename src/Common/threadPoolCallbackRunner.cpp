#include <Common/threadPoolCallbackRunner.h>

#include <Common/futex.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

void CountingSemaphore::acquire()
{
    UInt32 x = val.load(std::memory_order_relaxed);
    while (true)
    {
        if (x == 0)
        {
            futexWait(&val, 0);
            x = val.load(std::memory_order_relaxed);
        }
        else if (val.compare_exchange_weak(
                    x, x - 1, std::memory_order_acquire, std::memory_order_relaxed))
            break;
    }
}

void CountingSemaphore::release(UInt32 n, UInt32 wake_threshold)
{
    UInt32 x = val.fetch_add(n, std::memory_order_release);
    if (x < wake_threshold)
        futexWake(&val, n);
}

ThreadPoolCallbackRunnerFast::ThreadPoolCallbackRunnerFast() = default;

void ThreadPoolCallbackRunnerFast::initThreadPool(ThreadPool & pool_, size_t max_threads_, std::string thread_name_, ThreadGroupPtr thread_group_)
{
    chassert(!pool);
    mode = Mode::ThreadPool;
    pool = &pool_;
    max_threads = max_threads_;
    thread_name = thread_name_;
    thread_group = thread_group_;

    /// TODO [parquet]: Maybe adjust number of threads dynamically based on queue size.
    threads = max_threads;
    for (size_t i = 0; i < max_threads; ++i)
        pool->scheduleOrThrowOnError([this] { threadFunction(); });
}

ThreadPoolCallbackRunnerFast::ThreadPoolCallbackRunnerFast(Mode mode_) : mode(mode_)
{
    chassert(mode != Mode::ThreadPool);
}

ThreadPoolCallbackRunnerFast::~ThreadPoolCallbackRunnerFast()
{
    shutdown();
}

void ThreadPoolCallbackRunnerFast::shutdown()
{
    /// May be called twice.
    std::unique_lock lock(mutex);
    shutdown_requested = true;
    queue_sem.release(UINT32_MAX / 4);
    shutdown_cv.wait(lock, [&] { return threads == 0; });
}

void ThreadPoolCallbackRunnerFast::operator()(std::function<void()> f)
{
    if (mode == Mode::Disabled)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Thread pool runner is not initialized");

    {
        std::unique_lock lock(mutex);
        queue.push_back(std::move(f));
    }

    if (mode == Mode::ThreadPool)
        queue_sem.release(1, max_threads);
}

void ThreadPoolCallbackRunnerFast::bulkSchedule(std::vector<std::function<void()>> fs)
{
    if (fs.empty())
        return;

    if (mode == Mode::Disabled)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Thread pool runner is not initialized");

    {
        std::unique_lock lock(mutex);
        queue.insert(queue.end(), std::move_iterator(fs.begin()), std::move_iterator(fs.end()));
    }

    if (mode == Mode::ThreadPool)
        queue_sem.release(fs.size(), max_threads);
}

bool ThreadPoolCallbackRunnerFast::runTaskInline()
{
    std::function<void()> f;
    {
        std::unique_lock lock(mutex);
        if (queue.empty())
            return false;
        f = std::move(queue.front());
        queue.pop_front();
    }
    f();
    return true;
}

void ThreadPoolCallbackRunnerFast::threadFunction()
{
    ThreadGroupSwitcher switcher(thread_group, thread_name.c_str());

    while (true)
    {
        queue_sem.acquire();

        std::function<void()> f;
        {
            std::unique_lock lock(mutex);

            if (shutdown_requested)
            {
                threads -= 1;
                if (threads == 0)
                    shutdown_cv.notify_all();
                return;
            }

            chassert(!queue.empty());

            f = std::move(queue.front());
            queue.pop_front();
        }

        try
        {
            f();

            CurrentThread::updatePerformanceCountersIfNeeded();
        }
        catch (...)
        {
            tryLogCurrentException("FastThreadPool");
            chassert(false);
        }
    }
}

bool ShutdownHelper::try_lock_shared()
{
    Int64 n = val.fetch_add(1, std::memory_order_acquire) + 1;
    chassert(n != SHUTDOWN_START);
    if (n >= SHUTDOWN_START)
    {
        unlock_shared();
        return false;
    }
    return true;
}

void ShutdownHelper::unlock_shared()
{
    Int64 n = val.fetch_sub(1, std::memory_order_release) - 1;
    chassert(n >= 0);
    if (n == SHUTDOWN_START)
    {
        /// We're the last completed task. Add SHUTDOWN_END to indicate that no further waiting
        /// or cv notifying is needed, even though `val` can get briefly bumped up and down by
        /// unsuccessful try_lock_shared() calls.
        val.fetch_add(SHUTDOWN_END);
        {
            /// Lock and unlock the mutex. This may look weird, but this is usually (always?)
            /// required to avoid race conditions when combining condition_variable with atomics.
            ///
            /// In this case, the prevented race condition is:
            ///  1. unlock_shared() sees n == SHUTDOWN_START,
            ///  2. shutdown thread enters cv.wait(lock, [&] { return val.load() >= SHUTDOWN_END; });
            ///     the callback does val.load(), gets SHUTDOWN_START, and is about
            ///     to return false; at this point, the cv.wait call is not monitoring
            ///     condition_variable notifications (remember that cv.wait with callback is
            ///     equivalent to a wait without callback in a loop),
            ///  3. the unlock_shared() assigns `val` and calls cv.notify_all(), which does
            ///     nothing because no thread is blocked on the condition variable,
            ///  4. the cv.wait callback returns false; the wait goes back to sleep and never
            ///     wakes up.
            std::unique_lock lock(mutex);
        }
        cv.notify_all();
    }
}

bool ShutdownHelper::shutdown_requested()
{
    return val.load(std::memory_order_relaxed) >= SHUTDOWN_START;
}

void ShutdownHelper::begin_shutdown()
{
    Int64 n = val.fetch_add(SHUTDOWN_START) + SHUTDOWN_START;
    chassert(n < SHUTDOWN_START * 2); // shutdown requested more than once
    if (n == SHUTDOWN_START)
        val.fetch_add(SHUTDOWN_END);
}

void ShutdownHelper::wait_shutdown()
{
    std::unique_lock lock(mutex);
    cv.wait(lock, [&] { return val.load() >= SHUTDOWN_END; });
}

void ShutdownHelper::shutdown()
{
    begin_shutdown();
    wait_shutdown();
}

template ThreadPoolCallbackRunnerUnsafe<void> threadPoolCallbackRunnerUnsafe<void>(ThreadPool &, const std::string &);
template class ThreadPoolCallbackRunnerLocal<void>;

}
