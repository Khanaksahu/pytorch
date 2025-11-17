# Owner(s): ["module: PrivateUse1"]

import gc
import threading
import time

import torch

import torch_openreg  # noqa: F401
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestDeviceAllocator(TestCase):
    """Test cases for OpenRegDeviceAllocator functionality."""

    def setUp(self):
        """Reset memory state before each test."""
        # Force garbage collection to ensure clean state
        gc.collect()
        # Note: We can't directly reset allocator stats without C++ API,
        # but we can ensure tensors are properly released

    def test_basic_allocation(self):
        """Test basic memory allocation with various sizes."""
        # Small allocation
        x = torch.empty(100, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.numel(), 100)
        # Large allocation
        z = torch.empty(10000, device="openreg")
        self.assertEqual(z.device.type, "openreg")
        self.assertEqual(z.numel(), 10000)
        # Multi-dimensional allocation
        w = torch.empty(10, 20, 30, device="openreg")
        self.assertEqual(w.device.type, "openreg")
        self.assertEqual(w.shape, torch.Size([10, 20, 30]))

    def test_memory_lifecycle(self):
        """Test complete memory allocation and deallocation lifecycle."""
        # Allocate tensor
        x = torch.empty(1000, device="openreg")
        self.assertEqual(x.device.type, "openreg")

        # Explicitly delete tensor
        del x
        gc.collect()

        # Allocate again to ensure memory was freed
        y = torch.empty(1000, device="openreg")
        self.assertEqual(y.device.type, "openreg")
        del y
        gc.collect()

    def test_tensor_copy_operations(self):
        """Test memory operations during tensor copies."""
        # CPU to OpenReg
        cpu_tensor = torch.randn(100)
        openreg_tensor = cpu_tensor.to("openreg")
        self.assertEqual(openreg_tensor.device.type, "openreg")
        self.assertEqual(cpu_tensor.shape, openreg_tensor.shape)

        # OpenReg to CPU
        back_to_cpu = openreg_tensor.to("cpu")
        self.assertEqual(back_to_cpu.device.type, "cpu")
        self.assertTrue(torch.allclose(cpu_tensor, back_to_cpu))

        # OpenReg to OpenReg (clone)
        cloned = openreg_tensor.clone()
        self.assertEqual(cloned.device.type, "openreg")
        self.assertTrue(torch.allclose(openreg_tensor.cpu(), cloned.cpu()))

    def test_inplace_operations(self):
        """Test memory stability during inplace operations."""
        x = torch.ones(100, device="openreg")
        original_data_ptr = x.data_ptr()

        # Inplace addition
        x.add_(1)
        self.assertEqual(x.data_ptr(), original_data_ptr)
        self.assertTrue(torch.all(x == 2))

        # Inplace multiplication
        x.mul_(2)
        self.assertEqual(x.data_ptr(), original_data_ptr)
        self.assertTrue(torch.all(x == 4))

    def test_view_operations(self):
        """Test that views share memory correctly."""
        x = torch.randn(100, device="openreg")
        original_data_ptr = x.data_ptr()

        # Reshape view
        y = x.view(10, 10)
        self.assertEqual(y.data_ptr(), original_data_ptr)
        self.assertEqual(y.shape, torch.Size([10, 10]))

        # Slice view
        z = x[10:20]
        # Slices may have different data_ptr but should share storage
        self.assertEqual(z.numel(), 10)

    def test_different_dtypes(self):
        """Test allocation with different data types."""
        dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]

        for dtype in dtypes:
            x = torch.empty(100, dtype=dtype, device="openreg")
            self.assertEqual(x.device.type, "openreg")
            self.assertEqual(x.dtype, dtype)
            self.assertEqual(x.numel(), 100)

    def test_tensor_resize(self):
        """Test tensor resizing operations."""
        x = torch.empty(100, device="openreg")
        _ = x.data_ptr()

        # Resize to smaller size (should reuse storage)
        x.resize_(50)
        self.assertEqual(x.numel(), 50)
        # Storage should still be available

        # Resize to original size
        x.resize_(100)
        self.assertEqual(x.numel(), 100)

    def test_empty_cache_operation(self):
        """Test empty cache functionality."""
        # Allocate some tensors
        x = torch.empty(1000, device="openreg")
        y = torch.empty(2000, device="openreg")

        # Delete tensors
        del x, y
        gc.collect()

        # Note: OpenRegDeviceAllocator.emptyCache is currently a no-op
        # This test ensures it doesn't crash
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # def test_cross_device_allocation(self):
    #    """Test allocation on different device indices if supported."""
    # Test device 0
    #    x = torch.empty(100, device="openreg:0")
    #    self.assertEqual(x.device.type, "openreg")
    #    self.assertEqual(x.device.index, 0)

    # Test cross device
    #    y = torch.empty(100, device="openreg:1")
    #    self.assertEqual(y.device.type, "openreg")
    #    self.assertEqual(y.device.index, 1)

    def test_memory_format_allocation(self):
        """Test allocation with different memory formats."""
        # Channels last format
        x = torch.empty(2, 3, 4, 4, device="openreg", memory_format=torch.channels_last)
        self.assertEqual(x.device.type, "openreg")
        self.assertTrue(x.is_contiguous(memory_format=torch.channels_last))

        # Contiguous format (default)
        y = torch.empty(
            2, 3, 4, 4, device="openreg", memory_format=torch.contiguous_format
        )
        self.assertEqual(y.device.type, "openreg")
        self.assertTrue(y.is_contiguous())

    def test_large_allocation(self):
        """Test large memory allocation."""
        # Allocate a large tensor (10MB approximately)
        size = 10 * 1024 * 1024 // 4  # 10MB in float32
        x = torch.empty(size, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.numel(), size)

    def test_sequential_allocations_and_deallocations(self):
        """Test sequential allocation and deallocation patterns."""
        for i in range(10):
            x = torch.empty(1000 + i * 100, device="openreg")
            self.assertEqual(x.device.type, "openreg")
            # Let tensor go out of scope
            del x
        gc.collect()

    def test_allocation_with_requires_grad(self):
        """Test allocation of tensors with gradient tracking."""
        x = torch.empty(100, device="openreg", requires_grad=True)
        self.assertEqual(x.device.type, "openreg")
        self.assertTrue(x.requires_grad)

        y = torch.randn(100, device="openreg", requires_grad=True)
        self.assertEqual(y.device.type, "openreg")
        self.assertTrue(y.requires_grad)

    def test_storage_operations(self):
        """Test storage-level operations."""
        x = torch.randn(100, device="openreg")
        storage = x.storage()

        # Verify storage is on correct device
        self.assertTrue(storage.device.type == "openreg")

        # Verify storage size
        self.assertGreaterEqual(storage.size(), x.numel())

    def test_tensor_from_blob(self):
        """Test creating tensors that reference existing memory."""
        x = torch.randn(100, device="openreg")

        # Create a view that references the same data
        y = x.view_as(x)

        # They should share the same underlying storage
        self.assertEqual(x.data_ptr(), y.data_ptr())

        # Modifying one should affect the other
        x.fill_(5.0)
        self.assertTrue(torch.all(y == 5.0))


class TestMultithreadingSafety(TestCase):
    """Test cases for multithreading safety of OpenRegDeviceAllocator."""

    def setUp(self):
        """Reset memory state before each test."""
        gc.collect()

    def test_concurrent_allocations(self):
        """Test concurrent allocations from multiple threads."""
        num_threads = 10
        allocations_per_thread = 50
        errors = []

        def allocate_tensors(thread_id):
            try:
                for i in range(allocations_per_thread):
                    size = 100 + (thread_id * 10) + i
                    x = torch.empty(size, device="openreg")
                    self.assertEqual(x.device.type, "openreg")
                    self.assertEqual(x.numel(), size)
                    # Small delay to increase concurrency
                    time.sleep(0.001)
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=allocate_tensors, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        # Check no errors occurred
        self.assertEqual(len(errors), 0, f"Errors in threads: {errors}")

    def test_concurrent_allocations_and_deallocations(self):
        """Test concurrent allocations and deallocations."""
        num_threads = 8
        iterations = 30
        errors = []

        def alloc_dealloc_cycle(thread_id):
            try:
                tensors = []
                for i in range(iterations):
                    # Allocate
                    x = torch.empty(500 + i * 10, device="openreg")
                    self.assertEqual(x.device.type, "openreg")
                    tensors.append(x)
                    # Deallocate older tensors
                    if len(tensors) > 5:
                        tensors.pop(0)
                    time.sleep(0.001)
                # Clean up remaining tensors
                tensors.clear()
                gc.collect()
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=alloc_dealloc_cycle, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors in threads: {errors}")

    def test_concurrent_mixed_operations(self):
        """Test concurrent mixed operations: allocate, copy, modify, free."""
        num_threads = 6
        operations_per_thread = 40
        errors = []

        def mixed_operations(thread_id):
            try:
                for i in range(operations_per_thread):
                    # Allocate
                    x = torch.randn(200, device="openreg")

                    # Copy to CPU and back
                    cpu_copy = x.cpu()
                    back_to_device = cpu_copy.to("openreg")

                    # Inplace operations
                    back_to_device.add_(1)
                    back_to_device.mul_(2)

                    # Clone
                    cloned = back_to_device.clone()

                    # Verify correctness
                    self.assertEqual(cloned.device.type, "openreg")

                    time.sleep(0.001)

                    # Tensors go out of scope
                    del x, cpu_copy, back_to_device, cloned

                gc.collect()
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=mixed_operations, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors in threads: {errors}")

    def test_shared_tensor_across_threads(self):
        """Test accessing shared tensors from multiple threads (read-only)."""
        # Create a shared tensor
        shared_tensor = torch.randn(1000, device="openreg")
        shared_value = shared_tensor.cpu()

        num_threads = 8
        errors = []

        def read_shared_tensor(thread_id):
            try:
                for _ in range(50):
                    # Read from shared tensor
                    cpu_copy = shared_tensor.cpu()

                    # Verify it matches original
                    self.assertTrue(torch.allclose(cpu_copy, shared_value))

                    time.sleep(0.001)
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=read_shared_tensor, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors in threads: {errors}")

    def test_race_condition_stress(self):
        """Stress test for race conditions with rapid alloc/dealloc."""
        num_threads = 12
        rapid_cycles = 100
        errors = []

        def rapid_alloc_dealloc(thread_id):
            try:
                for i in range(rapid_cycles):
                    # Rapid allocation
                    x = torch.empty(100 + (i % 50), device="openreg")
                    # Immediate deallocation
                    del x

                    # No sleep - maximum concurrency stress

                gc.collect()
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=rapid_alloc_dealloc, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors in threads: {errors}")


class TestMemoryLeaks(TestCase):
    """Test cases for detecting memory leaks in OpenRegDeviceAllocator."""

    def setUp(self):
        """Reset memory state before each test."""
        gc.collect()
        time.sleep(0.1)  # Allow time for cleanup

    def test_no_leak_simple_allocations(self):
        """Test that simple allocations don't leak memory."""
        # Warm-up
        for _ in range(10):
            x = torch.empty(1000, device="openreg")
            del x
        gc.collect()
        time.sleep(0.1)

        # Perform many allocations and deallocations
        iterations = 1000
        for i in range(iterations):
            x = torch.empty(1000, device="openreg")
            del x

            if i % 100 == 0:
                gc.collect()

        # Final cleanup
        gc.collect()
        time.sleep(0.1)

        # If there were leaks, this would have accumulated significant memory
        # The test passes if no exception/crash occurred

    def test_no_leak_varying_sizes(self):
        """Test that allocations of varying sizes don't leak."""
        iterations = 500
        sizes = [100, 500, 1000, 5000, 10000]

        for i in range(iterations):
            size = sizes[i % len(sizes)]
            x = torch.empty(size, device="openreg")
            del x

            if i % 50 == 0:
                gc.collect()

        gc.collect()
        time.sleep(0.1)

    def test_no_leak_with_copies(self):
        """Test that tensor copies don't leak memory."""
        iterations = 300

        for i in range(iterations):
            # Create tensor
            x = torch.randn(500, device="openreg")

            # Copy to CPU
            cpu_copy = x.cpu()

            # Copy back to device
            device_copy = cpu_copy.to("openreg")

            # Clone
            cloned = device_copy.clone()

            # Delete all
            del x, cpu_copy, device_copy, cloned

            if i % 50 == 0:
                gc.collect()

        gc.collect()
        time.sleep(0.1)

    def test_no_leak_with_views(self):
        """Test that tensor views don't leak memory."""
        iterations = 500

        for i in range(iterations):
            x = torch.randn(1000, device="openreg")

            # Create various views
            view1 = x.view(10, 100)
            view2 = x[100:200]
            view3 = x.reshape(20, 50)

            # Delete views and original
            del view1, view2, view3, x

            if i % 100 == 0:
                gc.collect()

        gc.collect()
        time.sleep(0.1)

    def test_no_leak_inplace_operations(self):
        """Test that inplace operations don't leak memory."""
        iterations = 500

        for i in range(iterations):
            x = torch.ones(1000, device="openreg")

            # Multiple inplace operations
            x.add_(1)
            x.mul_(2)
            x.div_(2)
            x.sub_(1)

            del x

            if i % 100 == 0:
                gc.collect()

        gc.collect()
        time.sleep(0.1)

    def test_no_leak_with_gradients(self):
        """Test that tensors with gradients don't leak."""
        iterations = 300

        for i in range(iterations):
            x = torch.randn(100, device="openreg", requires_grad=True)
            y = torch.randn(100, device="openreg", requires_grad=True)

            # Operation that creates computation graph
            z = x + y

            # Delete all
            del x, y, z

            if i % 50 == 0:
                gc.collect()

        gc.collect()
        time.sleep(0.1)

    def test_no_leak_repeated_large_allocations(self):
        """Test repeated large allocations for memory leaks."""
        # Large tensor size (50MB)
        size = 50 * 1024 * 1024 // 4
        iterations = 50

        for i in range(iterations):
            x = torch.empty(size, device="openreg")
            del x
            gc.collect()
            time.sleep(0.05)  # Allow time for cleanup

        # Final cleanup
        gc.collect()
        time.sleep(0.1)

    def test_leak_detection_with_statistics(self):
        """Test memory leak detection using allocation patterns."""
        # This test verifies that after many alloc/dealloc cycles,
        # the allocator properly frees memory

        num_cycles = 10
        allocations_per_cycle = 100

        for cycle in range(num_cycles):
            tensors = []

            # Allocate many tensors
            for i in range(allocations_per_cycle):
                t = torch.empty(1000, device="openreg")
                tensors.append(t)

            # Verify all allocated
            self.assertEqual(len(tensors), allocations_per_cycle)

            # Delete all
            tensors.clear()
            gc.collect()
            time.sleep(0.05)

        # Final verification - if there were leaks, memory would be exhausted
        # The test passes if we can still allocate
        final_tensor = torch.empty(10000, device="openreg")
        self.assertEqual(final_tensor.device.type, "openreg")
        del final_tensor


class TestPinMemory(TestCase):
    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pin_memory(self):
        tensor = torch.randn(10)
        self.assertFalse(tensor.is_pinned())
        pinned_tensor = tensor.pin_memory()
        self.assertTrue(pinned_tensor.is_pinned())
        slice_tensor = pinned_tensor[2:5]
        self.assertTrue(slice_tensor.is_pinned())

        tensor = torch.randn(10)
        storage = tensor.storage()
        self.assertFalse(storage.is_pinned("openreg"))
        pinned_storage = storage.pin_memory("openreg")
        self.assertTrue(pinned_storage.is_pinned("openreg"))

        tensor = torch.randn(10)
        untyped_storage = tensor.untyped_storage()
        self.assertFalse(untyped_storage.is_pinned("openreg"))
        pinned_untyped_storage = untyped_storage.pin_memory("openreg")
        self.assertTrue(pinned_untyped_storage.is_pinned("openreg"))


class TestMultiDeviceAllocation(TestCase):
    """Test basic multi-device allocation functionality."""

    def setUp(self):
        self.device_count = torch.openreg.device_count()
        self.assertEqual(self.device_count, 2, "This test requires 2 OpenReg devices")
        gc.collect()

    def test_allocation_on_device_1(self):
        torch.openreg.set_device(1)
        x = torch.empty(100, device="openreg:1")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.device.index, 1)

    def test_device_switching_allocation(self):
        """Test allocation after device switching."""

        torch.openreg.set_device(0)
        x = torch.empty(100, device="openreg")
        self.assertEqual(x.device.index, 0)

        torch.openreg.set_device(1)
        y = torch.empty(100, device="openreg")
        self.assertEqual(y.device.index, 1)

        torch.openreg.set_device(0)
        z = torch.empty(100, device="openreg")
        self.assertEqual(z.device.index, 0)

    def test_simultaneous_device_allocations(self):
        """Test allocations on both devices simultaneously."""
        x = torch.empty(100, device="openreg:0")
        y = torch.empty(200, device="openreg:1")

        self.assertEqual(x.device.index, 0)
        self.assertEqual(y.device.index, 1)
        self.assertNotEqual(x.data_ptr(), y.data_ptr())

    def test_memory_isolation_between_devices(self):
        """Test that memory allocations are isolated between devices."""

        tensors_dev0 = [torch.empty(1000, device="openreg:0") for _ in range(10)]
        tensors_dev1 = [torch.empty(1000, device="openreg:1") for _ in range(10)]

        # Verify all device 0 tensors are on device 0
        for t in tensors_dev0:
            self.assertEqual(t.device.index, 0)

        # Verify all device 1 tensors are on device 1
        for t in tensors_dev1:
            self.assertEqual(t.device.index, 1)

        # Pointers should be different
        ptrs_dev0 = {t.data_ptr() for t in tensors_dev0}
        ptrs_dev1 = {t.data_ptr() for t in tensors_dev1}
        self.assertEqual(
            len(ptrs_dev0 & ptrs_dev1), 0, "Devices should not share pointers"
        )

    def test_alternating_device_allocations(self):
        """Test alternating allocations between devices."""
        tensors = []
        for i in range(20):
            device_idx = i % 2
            t = torch.empty(100 + i, device=f"openreg:{device_idx}")
            self.assertEqual(t.device.index, device_idx)
            tensors.append(t)

        # Verify all tensors retained correct device assignment
        for i, t in enumerate(tensors):
            expected_device = i % 2
            self.assertEqual(t.device.index, expected_device)


class TestCrossDeviceOperations(TestCase):
    """Test cross-device tensor operations."""

    def setUp(self):
        self.device_count = torch.openreg.device_count()
        self.assertEqual(self.device_count, 2)
        gc.collect()

    def test_tensor_to_different_device(self):
        """Test moving tensor from one device to another."""
        # Create on device 0
        x = torch.randn(100, device="openreg:0")
        self.assertEqual(x.device.index, 0)

        # Move to device 1
        y = x.to("openreg:1")
        self.assertEqual(y.device.index, 1)
        self.assertNotEqual(x.data_ptr(), y.data_ptr())

        # Values should be the same
        self.assertTrue(torch.allclose(x.cpu(), y.cpu()))

    def test_bidirectional_device_transfer(self):
        """Test transferring tensor back and forth between devices."""
        original = torch.randn(100, device="openreg:0")
        original_cpu = original.cpu()

        # 0 -> 1
        on_dev1 = original.to("openreg:1")
        self.assertTrue(torch.allclose(original_cpu, on_dev1.cpu()))

        # 1 -> 0
        back_to_dev0 = on_dev1.to("openreg:0")
        self.assertTrue(torch.allclose(original_cpu, back_to_dev0.cpu()))


class TestMultiDeviceConcurrency(TestCase):
    """Test concurrent operations on multiple devices."""

    def setUp(self):
        self.device_count = torch.openreg.device_count()
        self.assertEqual(self.device_count, 2)
        gc.collect()

    def test_concurrent_allocation_on_different_devices(self):
        """Test concurrent allocations on device 0 and device 1."""
        errors = []

        def allocate_on_device(device_idx, num_allocs):
            try:
                torch.openreg.set_device(device_idx)
                for i in range(num_allocs):
                    x = torch.empty(100 + i, device=f"openreg:{device_idx}")
                    self.assertEqual(x.device.index, device_idx)
                    time.sleep(0.001)
            except Exception as e:
                errors.append((device_idx, str(e)))

        threads = []
        for dev_idx in range(2):
            t = threading.Thread(target=allocate_on_device, args=(dev_idx, 50))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")

    def test_device_switching_race_condition(self):
        """Test race conditions when switching devices."""
        errors = []

        def switch_and_allocate(thread_id):
            try:
                for i in range(30):
                    device_idx = i % 2
                    torch.openreg.set_device(device_idx)
                    x = torch.empty(100, device="openreg")
                    self.assertEqual(x.device.index, device_idx)
                    time.sleep(0.001)
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(4):
            t = threading.Thread(target=switch_and_allocate, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")

    def test_concurrent_cross_device_copies(self):
        """Test concurrent copies between devices."""
        errors = []

        def copy_between_devices(thread_id, iterations):
            try:
                for i in range(iterations):
                    # Device 0 -> Device 1
                    x = torch.randn(100, device="openreg:0")
                    y = x.to("openreg:1")
                    self.assertTrue(torch.allclose(x.cpu(), y.cpu()))

                    # Device 1 -> Device 0
                    z = y.to("openreg:0")
                    self.assertTrue(torch.allclose(y.cpu(), z.cpu()))

                    time.sleep(0.001)
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(4):
            t = threading.Thread(target=copy_between_devices, args=(i, 20))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")

    def test_concurrent_mixed_device_operations(self):
        """Test concurrent mixed operations across devices."""
        errors = []

        def mixed_device_operations(thread_id, iterations):
            try:
                for i in range(iterations):
                    # Allocate on both devices
                    x0 = torch.randn(50, device="openreg:0")
                    x1 = torch.randn(50, device="openreg:1")

                    # Cross-device copies
                    x0_to_1 = x0.to("openreg:1")
                    x1_to_0 = x1.to("openreg:0")

                    # CPU copies
                    x0_cpu = x0.cpu()
                    x1_cpu = x1.cpu()

                    # Verify
                    self.assertTrue(torch.allclose(x0_cpu, x0_to_1.cpu()))
                    self.assertTrue(torch.allclose(x1_cpu, x1_to_0.cpu()))

                    time.sleep(0.001)
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(6):
            t = threading.Thread(target=mixed_device_operations, args=(i, 15))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")


if __name__ == "__main__":
    run_tests()
