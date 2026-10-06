//! Safe wrapper for the Qenivo C ABI.
//!
//! `simplex` / `native` call the revised simplex in `simplex_core.cpp`.
//! Any other engine name is solved by the embedded Python bridge.
//! Link with `libqenivo_c` (see `QENIVO_C_DLL` only for languages that load at
//! runtime; this crate links the library in the usual way).

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

#[repr(C)]
struct QenivoModel {
    _private: [u8; 0],
}

#[link(name = "qenivo_c")]
extern "C" {
    fn qenivo_model_create(
        rows: i32, cols: i32, a_rowptr: *const i32, a_colidx: *const i32, a_values: *const f64,
        cost: *const f64, row_lower: *const f64, row_upper: *const f64, col_lower: *const f64,
        col_upper: *const f64, integrality: *const i32, q_rowptr: *const i32, q_colidx: *const i32,
        q_values: *const f64, error: *mut i32,
    ) -> *mut QenivoModel;
    fn qenivo_model_free(model: *mut QenivoModel);
    fn qenivo_create_error_message() -> *const c_char;
    fn qenivo_last_error_message(model: *const QenivoModel) -> *const c_char;
    fn qenivo_solve(model: *mut QenivoModel, engine: *const c_char) -> i32;
    fn qenivo_result_status(model: *const QenivoModel) -> i32;
    fn qenivo_result_objective(model: *const QenivoModel, objective: *mut f64) -> i32;
    fn qenivo_result_x(model: *const QenivoModel, x: *mut f64, cols: i32) -> i32;
}

fn c_str(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

pub struct Solution {
    pub status: i32,
    pub objective: f64,
}

pub struct Model {
    handle: *mut QenivoModel,
}

impl Model {
    pub fn create(
        rows: i32, cost: &[f64], row_ptr: &[i32], col_idx: &[i32], values: &[f64],
        row_lower: &[f64], row_upper: &[f64], col_lower: &[f64], col_upper: &[f64],
        integrality: Option<&[i32]>,
    ) -> Result<Self, String> {
        let mut err = 0i32;
        let handle = unsafe {
            qenivo_model_create(
                rows,
                cost.len() as i32,
                row_ptr.as_ptr(),
                if col_idx.is_empty() { ptr::null() } else { col_idx.as_ptr() },
                if values.is_empty() { ptr::null() } else { values.as_ptr() },
                cost.as_ptr(),
                if row_lower.is_empty() { ptr::null() } else { row_lower.as_ptr() },
                if row_upper.is_empty() { ptr::null() } else { row_upper.as_ptr() },
                col_lower.as_ptr(),
                col_upper.as_ptr(),
                integrality.map(|v| v.as_ptr()).unwrap_or(ptr::null()),
                ptr::null(), ptr::null(), ptr::null(),
                &mut err,
            )
        };
        if handle.is_null() {
            return Err(format!("create failed ({err}): {}", unsafe { c_str(qenivo_create_error_message()) }));
        }
        Ok(Model { handle })
    }

    pub fn solve(&self, engine: &str) -> Result<Solution, String> {
        let name = CString::new(engine).map_err(|_| "engine name contains a nul")?;
        let rc = unsafe { qenivo_solve(self.handle, name.as_ptr()) };
        if rc != 0 {
            return Err(unsafe { c_str(qenivo_last_error_message(self.handle)) });
        }
        let mut objective = 0.0;
        if unsafe { qenivo_result_objective(self.handle, &mut objective) } != 0 {
            return Err("no objective".into());
        }
        let status = unsafe { qenivo_result_status(self.handle) };
        Ok(Solution { status, objective })
    }

    pub fn primal(&self, cols: usize) -> Result<Vec<f64>, String> {
        let mut x = vec![0.0; cols];
        if unsafe { qenivo_result_x(self.handle, x.as_mut_ptr(), cols as i32) } != 0 {
            return Err("no primal solution".into());
        }
        Ok(x)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { qenivo_model_free(self.handle) }
    }
}

/// min -x subject to 0 <= x <= 1. Returns the objective, which is -1.
pub fn bounded_lp() -> Result<f64, String> {
    let model = Model::create(0, &[-1.0], &[0], &[], &[], &[], &[], &[0.0], &[1.0], None)?;
    let sol = model.solve("simplex")?;
    let x = model.primal(1)?;
    if (sol.objective + 1.0).abs() > 1e-8 || (x[0] - 1.0).abs() > 1e-8 {
        return Err(format!("expected optimum -1 at x=1, got {} at {}", sol.objective, x[0]));
    }
    Ok(sol.objective)
}

/// min -x-2y s.t. x+y <= 1, x,y binary. Returns the objective, which is -2.
pub fn two_binary() -> Result<f64, String> {
    let model = Model::create(
        1,
        &[-1.0, -2.0],
        &[0, 2],
        &[0, 1],
        &[1.0, 1.0],
        &[f64::NEG_INFINITY],
        &[1.0],
        &[0.0, 0.0],
        &[1.0, 1.0],
        Some(&[1, 1]),
    )?;
    let sol = model.solve("milp")?;
    let x = model.primal(2)?;
    if (sol.objective + 2.0).abs() > 1e-6 || x[0].abs() > 1e-6 || (x[1] - 1.0).abs() > 1e-6 {
        return Err(format!("expected -2 at (0,1), got {} at {:?}", sol.objective, x));
    }
    Ok(sol.objective)
}
