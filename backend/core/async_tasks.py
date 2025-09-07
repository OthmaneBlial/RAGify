import asyncio
import logging
import inspect
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


@dataclass
class TaskProgress:
    """Progress tracking for tasks."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    current_step: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def percentage(self) -> float:
        """Calculate completion percentage."""
        return (self.completed / self.total * 100) if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "current_step": self.current_step,
            "percentage": self.percentage,
            "details": self.details,
        }


@dataclass
class Task:
    """Represents an async task."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    func: Callable = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    progress: TaskProgress = field(default_factory=TaskProgress)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    result: Any = None
    callback: Optional[Callable] = None

    def __lt__(self, other: "Task") -> bool:
        """Priority comparison for queue."""
        return self.priority.value < other.priority.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority.value,
            "status": self.status.value,
            "progress": self.progress.to_dict(),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "error_message": self.error_message,
        }


class AsyncTaskManager:
    """Manager for async task processing with priority queue."""

    def __init__(self, max_workers: int = 4, max_queue_size: int = 1000):
        self.queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.tasks: Dict[str, Task] = {}
        self.max_workers = max_workers
        self.workers: List[asyncio.Task] = []
        self.running = False
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="task-worker"
        )

    async def start(self):
        """Start the task manager."""
        if self.running:
            return

        self.running = True
        logger.info(f"Starting AsyncTaskManager with {self.max_workers} workers")

        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self.workers.append(worker)

    async def stop(self):
        """Stop the task manager."""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping AsyncTaskManager")

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)

        # Shutdown executor
        self.executor.shutdown(wait=True)

    async def submit_task(
        self,
        name: str,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        callback: Optional[Callable] = None,
        **kwargs,
    ) -> str:
        """
        Submit a task for execution.

        Args:
            name: Task name
            func: Function to execute
            priority: Task priority
            max_retries: Maximum retry attempts
            callback: Callback function for completion
            *args, **kwargs: Function arguments

        Returns:
            Task ID
        """
        task = Task(
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            callback=callback,
        )

        self.tasks[task.id] = task

        try:
            await self.queue.put(task)
            logger.info(f"Task submitted: {task.id} - {name}")
        except asyncio.QueueFull:
            logger.error(f"Task queue full, cannot submit: {name}")
            raise RuntimeError("Task queue is full")

        return task.id

    async def _worker_loop(self, worker_id: int):
        """Worker loop for processing tasks."""
        logger.info(f"Worker {worker_id} started")

        while self.running:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._execute_task(task)
            except Exception as e:
                logger.error(f"Worker {worker_id} error executing task {task.id}: {e}")
            finally:
                self.queue.task_done()

        logger.info(f"Worker {worker_id} stopped")

    async def _execute_task(self, task: Task):
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        try:
            if inspect.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor, partial(task.func, *task.args, **task.kwargs)
                )
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
        except Exception as e:
            await self._handle_task_error(task, str(e))
        if task.callback:
            try:
                await task.callback(task)
            except Exception as e:
                logger.error(f"Error in task callback for {task.id}: {e}")

    async def _handle_task_error(self, task: Task, error: str):
        """Handle task execution error."""
        task.error_message = error
        task.retry_count += 1

        if task.retry_count < task.max_retries:
            # Retry the task
            task.status = TaskStatus.RETRYING
            logger.warning(
                f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries}): {error}"
            )

            # Put back in queue with lower priority
            retry_task = Task(
                id=task.id,
                name=task.name,
                func=task.func,
                args=task.args,
                kwargs=task.kwargs,
                priority=TaskPriority(
                    min(task.priority.value + 1, TaskPriority.LOW.value)
                ),
                status=TaskStatus.PENDING,
                progress=task.progress,
                created_at=task.created_at,
                retry_count=task.retry_count,
                max_retries=task.max_retries,
                callback=task.callback,
            )

            await asyncio.sleep(2**task.retry_count)  # Exponential backoff
            await self.queue.put(retry_task)
        else:
            # Mark as failed
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            logger.error(f"Task {task.id} failed permanently: {error}")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status by ID."""
        task = self.tasks.get(task_id)
        return task.to_dict() if task else None

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks."""
        return [task.to_dict() for task in self.tasks.values()]

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            return True
        return False

    async def update_task_progress(
        self,
        task_id: str,
        completed: int = None,
        total: int = None,
        current_step: str = None,
        details: Dict[str, Any] = None,
    ):
        """Update task progress."""
        task = self.tasks.get(task_id)
        if task:
            if completed is not None:
                task.progress.completed = completed
            if total is not None:
                task.progress.total = total
            if current_step is not None:
                task.progress.current_step = current_step
            if details is not None:
                task.progress.details.update(details)


# Global task manager instance
task_manager = AsyncTaskManager()


# Convenience functions for common tasks
async def submit_document_processing_task(
    document_id: str, priority: TaskPriority = TaskPriority.NORMAL
) -> str:
    """Submit document processing task."""
    from ..modules.knowledge.processing import processing_service
    from ..core.database import get_db

    async def process_document():
        async for db in get_db():
            # Get document and process
            from ..modules.knowledge.crud import get_document

            document = await get_document(db, document_id)
            if document:
                result = await processing_service.process_document_after_creation(
                    document, db
                )
                return result
            else:
                raise ValueError(f"Document {document_id} not found")

    return await task_manager.submit_task(
        f"Process Document {document_id}", process_document, priority=priority
    )


async def submit_embedding_generation_task(
    document_id: str, priority: TaskPriority = TaskPriority.NORMAL
) -> str:
    """Submit embedding generation task."""
    from ..modules.rag.embedding_service import generate_document_embeddings
    from ..core.database import get_db

    async def generate_embeddings():
        async for db in get_db():
            result = await generate_document_embeddings(document_id, db)
            return result

    return await task_manager.submit_task(
        f"Generate Embeddings {document_id}", generate_embeddings, priority=priority
    )


async def submit_batch_processing_task(
    document_ids: List[str], priority: TaskPriority = TaskPriority.NORMAL
) -> str:
    """Submit batch document processing task."""
    from ..modules.knowledge.processing import processing_service
    from ..core.database import get_db

    async def process_batch():
        async for db in get_db():
            # Get documents
            from ..modules.knowledge.crud import get_documents_by_ids

            documents = await get_documents_by_ids(db, document_ids)
            if documents:
                results = await processing_service.batch_process_documents(
                    documents, db
                )
                return results
            else:
                raise ValueError("No documents found for batch processing")

    return await task_manager.submit_task(
        f"Batch Process {len(document_ids)} Documents", process_batch, priority=priority
    )


# Task progress callback example
async def log_task_progress(task: Task):
    """Example callback to log task progress."""
    if task.status == TaskStatus.COMPLETED:
        logger.info(f"Task {task.id} completed with result: {task.result}")
    elif task.status == TaskStatus.FAILED:
        logger.error(f"Task {task.id} failed: {task.error_message}")
    else:
        progress = task.progress
        logger.info(
            f"Task {task.id} progress: {progress.percentage:.1f}% - {progress.current_step}"
        )
