// use openvino;

// fn main() {
//     assert!(openvino::version().build_number.starts_with("2"));
//     println!("OPENVINO_HOME: {:?}", std::env::var("OPENVINO_HOME"));
//     println!("OPENVINO_BUILD_DIR: {:?}", std::env::var("OPENVINO_BUILD_DIR"));
//     let test: openvino::Core = openvino::Core::new().expect("to instantiate the OpenVINO library");
//     let devices: Result<Vec<openvino::DeviceType<'_>>, openvino::InferenceError> = test.available_devices();
//     println!("Available devices: {:#?}", devices);
// }


use std::sync::Arc;//{, Mutex, MutexGuard};
use anyhow::{Context, Error, Ok, Result};
use onnxruntime::{
    environment::Environment,
    session::Session,
    tensor::OrtOwnedTensor,
    GraphOptimizationLevel
};
use ndarray_stats::QuantileExt;
use ndarray::{Array, s};

#[derive(Debug)]
// pub struct ModelWrapper<'a> {
//     session: Session<'a>,
//     environment: Arc<Mutex<onnxruntime::environment::Environment>>, // Store the environment in the struct
// }

pub struct ModelWrapper {
    environment: Arc<Environment>,
    session: Session<'static>,
}

impl ModelWrapper {
    pub fn new(model_path: String) -> Result<Self, Error> {
        let environment = Arc::new(Environment::builder()
            .with_name("phi-1-5")
            // .with_log_level(LoggingLevel::Verbose)
            .build()?);

        let environment_ref: &'static Environment = unsafe { &*(Arc::as_ptr(&environment) as *const Environment) };

        let session = environment_ref
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_number_threads(12)?
            .with_model_from_file(model_path)
            .map_err(|e| anyhow::anyhow!(e))?; // Convert OrtError to anyhow::Error

        Ok(Self {
            environment,
            session,
        })
    }

    pub fn prediction(&mut self, tokens: Vec<i64>) -> Result<i64, Error> {
        // Prepare the input tensor
        let input_array_i64 = Array::from_shape_vec((1, tokens.len()), tokens.clone())?;

        // Convert i64 array to i32 array
        // let input_array_i32  = input_array_i64.mapv(|x| x as i32);

        // Run the session with the input tensor
        let outputs: Vec<OrtOwnedTensor<f32, ndarray::Dim<ndarray::IxDynImpl>>> = self.session.run(vec![input_array_i64])?;
        // let reshaped_output = outputs[0];
        let prediction: i64 = i64::try_from(
            QuantileExt::argmax(
                &outputs[0]
                    .slice(s![0, -1, ..])
                    .into_shape([151936])
                    .unwrap()
            ).context("Argmax failed")?
        ).expect("Conversion to i64 failed");
        // Process the output
        // println!("{:?}", prediction);
        Ok(prediction)
    }
}