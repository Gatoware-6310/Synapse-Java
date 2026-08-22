package xyz.gatoware.synapse;

import xyz.gatoware.synapse.backend.Backend;
import xyz.gatoware.synapse.backend.CpuBackend;
import xyz.gatoware.synapse.backend.CudaBackend;

/** Global Synapse runtime configuration. */
public final class Synapse {
	private static volatile Devices device = Devices.CPU;
	private static volatile Backend backend = CpuBackend.INSTANCE;

	private Synapse() {
	}

	/** Selects the compute device used by supported Synapse operations.
	 * CPU is used by default when this method is never called.
	 * @param newDevice the device to use
	 * @throws IllegalArgumentException if newDevice is null
	 * @throws IllegalStateException if CUDA is requested but unavailable
	 */
	public static synchronized void useDevice(Devices newDevice) {
		if (newDevice == null)
			throw new IllegalArgumentException("Device cannot be null");
		if (newDevice == device)
			return;

		Backend nextBackend = switch (newDevice) {
			case CPU -> CpuBackend.INSTANCE;
			case CUDA -> new CudaBackend();
		};

		Backend previousBackend = backend;
		backend = nextBackend;
		device = newDevice;
		if (previousBackend != CpuBackend.INSTANCE)
			previousBackend.close();
	}

	/** Returns the currently selected compute device.
	 * @return the active device
	 */
	public static Devices getDevice() {
		return device;
	}

	/** Checks whether a device is available on this system.
	 * @param requestedDevice the device to check
	 * @return whether the device is available
	 */
	public static boolean isDeviceAvailable(Devices requestedDevice) {
		if (requestedDevice == null)
			return false;
		return switch (requestedDevice) {
			case CPU -> true;
			case CUDA -> CudaBackend.isAvailable();
		};
	}

	/** Returns the active internal compute backend.
	 * This is public so Synapse subpackages can share the selected backend;
	 * ordinary library users should prefer useDevice(Devices).
	 * @return the active backend
	 */
	public static Backend backend() {
		return backend;
	}
}
