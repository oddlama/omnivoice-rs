use std::{env, fs, path::PathBuf};

use fslock::LockFile;

use crate::{OmniVoiceError, Result};

#[derive(Debug)]
pub struct GpuTestLockGuard {
    lock: LockFile,
}

impl Drop for GpuTestLockGuard {
    fn drop(&mut self) {
        let _ = self.lock.unlock();
    }
}

pub fn acquire_gpu_test_lock() -> Result<GpuTestLockGuard> {
    let lock_path = gpu_test_lock_path();
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut lock = LockFile::open(&lock_path).map_err(OmniVoiceError::Io)?;
    lock.lock().map_err(OmniVoiceError::Io)?;
    Ok(GpuTestLockGuard { lock })
}

fn gpu_test_lock_path() -> PathBuf {
    env::var_os("OMNIVOICE_GPU_TEST_LOCK")
        .map(PathBuf::from)
        .unwrap_or_else(|| env::temp_dir().join("omnivoice-gpu-tests.lock"))
}
