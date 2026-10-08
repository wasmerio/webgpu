//! The import surface is the trust boundary: whatever a guest passes, the host
//! must answer with a value or a trap, never a panic that escapes or memory
//! unsafety.
//!
//! This calls every import with all-zero arguments and with all-ones
//! arguments (null and garbage handles, null and wild pointers, absurd
//! counts and sizes) from a plain wasm module, with no shim in between.

use wasmer::{AsStoreMut, Instance, Module, Store, Value};
use wasmer_webgpu::WebGpuCtx;
use wasmer_webgpu::abi::{IMPORTS, ValType};

fn wat_type(ty: &ValType) -> &'static str {
    match ty {
        ValType::I32 => "i32",
        ValType::I64 => "i64",
        ValType::F32 => "f32",
        ValType::F64 => "f64",
    }
}

fn constant(ty: &ValType, ones: bool) -> String {
    match (ty, ones) {
        (ValType::I32, false) => "(i32.const 0)".into(),
        (ValType::I32, true) => "(i32.const -1)".into(),
        (ValType::I64, false) => "(i64.const 0)".into(),
        (ValType::I64, true) => "(i64.const -1)".into(),
        (ValType::F32, false) => "(f32.const 0)".into(),
        (ValType::F32, true) => "(f32.const nan)".into(),
        (ValType::F64, false) => "(f64.const 0)".into(),
        (ValType::F64, true) => "(f64.const nan)".into(),
    }
}

#[test]
fn every_import_survives_null_and_garbage_arguments() {
    let mut wat = String::from("(module\n");
    for (index, sig) in IMPORTS.iter().enumerate() {
        let params: Vec<_> = sig.params.iter().map(wat_type).collect();
        let results: Vec<_> = sig.results.iter().map(wat_type).collect();
        wat.push_str(&format!(
            "(import \"{}\" \"{}\" (func $f{index} (param {}) (result {})))\n",
            wasmer_webgpu::WEBGPU_MODULE_NAME,
            sig.name,
            params.join(" "),
            results.join(" ")
        ));
    }
    wat.push_str("(memory (export \"memory\") 1)\n");
    for (index, sig) in IMPORTS.iter().enumerate() {
        for (suffix, ones) in [("zeros", false), ("ones", true)] {
            let args: Vec<_> = sig.params.iter().map(|ty| constant(ty, ones)).collect();
            // Results are dropped: only surviving the call matters here.
            let drop = if sig.results.is_empty() { "" } else { "drop" };
            wat.push_str(&format!(
                "(func (export \"{suffix}_{index}\") (call $f{index} {}) {drop})\n",
                args.join(" ")
            ));
        }
    }
    wat.push(')');

    let mut store = Store::default();
    let module = Module::new(&store, wat::parse_str(&wat).expect("wat parses")).expect("compiles");
    let hooks = WebGpuCtx::default().runtime_hooks();
    let (imports, state) = hooks
        .additional_imports(&module, &mut store.as_store_mut())
        .expect("imports register");
    let instance = Instance::new(&mut store, &module, &imports).expect("instantiates");
    hooks
        .configure_instance(&module, &mut store.as_store_mut(), &instance, None, state)
        .expect("configures");

    let mut trapped = 0;
    for round in 0..2 {
        for suffix in ["zeros", "ones"] {
            for (index, sig) in IMPORTS.iter().enumerate() {
                let function = instance
                    .exports
                    .get_function(&format!("{suffix}_{index}"))
                    .expect("export exists");
                // A trap is a fine answer to a wild pointer. A panic would
                // unwind out of `call` and fail this test.
                match function.call(&mut store, &[] as &[Value]) {
                    Ok(_) => {}
                    Err(error) => {
                        trapped += 1;
                        assert!(
                            error.message().contains(sig.name),
                            "round {round}: {} trapped without naming itself: {}",
                            sig.name,
                            error.message()
                        );
                    }
                }
            }
        }
    }
    // Wild pointers do trap; most garbage is simply refused.
    assert!(trapped > 0, "no import trapped on a wild pointer");
    println!(
        "{} imports called 4 times each, {trapped} traps",
        IMPORTS.len()
    );
}

/// A call that has a status to return could report a wild out-pointer as a
/// failure. It must not: the guest's own bug would then look like the GPU's.
#[test]
fn a_wild_out_pointer_traps_even_where_a_status_could_be_returned() {
    let wat = format!(
        r#"(module
            (import "{module}" "wgpuGetInstanceLimits" (func $limits (param i32) (result i32)))
            (memory (export "memory") 1)
            (func (export "limits") (param i32) (result i32) (call $limits (local.get 0))))"#,
        module = wasmer_webgpu::WEBGPU_MODULE_NAME
    );
    let mut store = Store::default();
    let module = Module::new(&store, wat::parse_str(&wat).expect("wat parses")).expect("compiles");
    let hooks = WebGpuCtx::default().runtime_hooks();
    let (imports, state) = hooks
        .additional_imports(&module, &mut store.as_store_mut())
        .expect("imports register");
    let instance = Instance::new(&mut store, &module, &imports).expect("instantiates");
    hooks
        .configure_instance(&module, &mut store.as_store_mut(), &instance, None, state)
        .expect("configures");
    let limits = instance
        .exports
        .get_function("limits")
        .expect("export exists");

    // Inside the memory: WGPUStatus_Success.
    let status = limits
        .call(&mut store, &[Value::I32(1024)])
        .expect("a valid pointer is served");
    assert_eq!(status[0].unwrap_i32(), 1);

    // One page is 64 KiB; this struct would end past it.
    for pointer in [65_532, -16] {
        let error = limits
            .call(&mut store, &[Value::I32(pointer)])
            .expect_err("a pointer outside the memory traps");
        assert!(
            error.message().contains("outside guest memory"),
            "{}",
            error.message()
        );
    }
}
