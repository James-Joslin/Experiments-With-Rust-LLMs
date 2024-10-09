// use std::sync::{Arc, Mutex};
// use anyhow::{Context, Error, Result};
// use onnxruntime::{
//     environment::Environment, 
//     session::Session, 
//     tensor::OrtOwnedTensor, 
//     GraphOptimizationLevel, 
//     LoggingLevel
// };
// use ndarray_stats::QuantileExt;
// use onnxruntime::ndarray::Array1;
// use ndarray::{
//     Array,
//     s
// };

// pub struct ModelWrapper {
//     environment: Arc<Environment>,
//     session: Arc<Mutex<Session<'static>>>,
// }

// impl<'a> ModelWrapper -> Result<> {
//     pub fn new(model_path: String) -> Result<Self, Error> {
//         let environment = Arc::new(Environment::builder()
//             .with_name("test")
//             .with_log_level(LoggingLevel::Verbose)
//             .build()?);

//         let session = environment
//             .new_session_builder()?
//             .with_optimization_level(GraphOptimizationLevel::All)?
//             .with_number_threads(8)?
//             .with_model_from_file(model_path)
//             .map_err(|e| anyhow::anyhow!(e))?; // Convert OrtError to anyhow::Error

//         Ok(Self {
//             environment,
//             session: Arc::new(Mutex::new(session)),
//         })
//     }


//     pub fn prediction(self, tokens: Vec<i64>) -> Result<i64, Error> {
//         // Prepare the input tensor
//         let input_array = Array::from_shape_vec((1, tokens.len()), tokens.clone())?;
//         // Run the session with the input tensor
//         let outputs: Vec<OrtOwnedTensor<f32, ndarray::Dim<ndarray::IxDynImpl>>> = self.session.run(vec![input_array])?;
//         // let reshaped_output = outputs[0];
//         let prediction: i64 = i64::try_from(
//             QuantileExt::argmax(
//                 &outputs[0]
//                     .slice(s![.., -1, ..])
//                     .into_shape([51200])
//                     .unwrap()
//             ).context("Argmax failed")?
//         ).expect("Conversion to i64 failed");
//         // Process the output
//         println!("{:?}", prediction);
//         Ok(prediction)
//     }
// }
