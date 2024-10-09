mod llm;
use llm::tokeniser_wrapper::TokenizerWrapper;
use llm::model_wrapper::ModelWrapper;
mod model;
use model::conversation::{self, Conversation};

slint::include_modules!();
use slint::{ModelRc, VecModel};

// use std::{thread, time};
use anyhow::{Error, Result};
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
// use std::sync::mpsc;
// use std::thread;
// use onnxruntime::{environment::Environment, LoggingLevel};//, tensor::OrtOwnedTensor};

impl From<conversation::Message> for slint_generatedAppWindow::SlintMessage {
    fn from(msg: conversation::Message) -> Self {
        slint_generatedAppWindow::SlintMessage {
            is_user: msg.is_user,
            text: msg.text.into(),
        }
    }
}

#[derive(Clone, Debug)]
struct LlmModel {
    inference : Arc<Mutex<ModelWrapper>>,
    tokens: Arc<Mutex<TokenizerWrapper>>,
    conversation : Arc<Mutex<Conversation>>
}


fn handle_send_chat_to_backend(ui_weak: slint::Weak<AppWindow>, llm_model: LlmModel, text: String, future_tokens: i16) {
    if let Some(ui) = ui_weak.upgrade() {
        let inference: Arc<Mutex<ModelWrapper>> = llm_model.inference.clone();//.lock().unwrap();.lock().unwrap();
        let tokens: Arc<Mutex<TokenizerWrapper>>  = llm_model.tokens.clone();//.lock().unwrap();
        let conversation: Arc<Mutex<Conversation>> = llm_model.conversation.clone();//.lock().unwrap();.lock().unwrap();

        // let ui_clone = Arc::new(Mutex::new(ui_weak.clone()));
        
        conversation.lock().unwrap().add_user_prompt(text.clone());

        let input_text: String = format!("{}Answer:\n", text.clone());

        tokens.lock().unwrap().store_initial_tokenize(&input_text.clone());

        conversation.lock().unwrap().add_ai_response(&"".to_string());

        for _ in 0..future_tokens {

            let prediction: i64 = inference.lock().unwrap().prediction(
                tokens.lock().unwrap().get_tokens_for_model()
            ).unwrap();
            
            tokens.lock().unwrap().append_token(prediction);

            if tokens.lock().unwrap().check_exit(prediction) {
                break;
            }
            
            let decoded_token: String = tokens.lock().unwrap().decode_token(prediction).unwrap();
            
            conversation.lock().unwrap().update_ai_response(decoded_token.clone());
            
            print!("{}", decoded_token.clone());
            io::stdout().flush().unwrap();

        }
        let messages_rc = ModelRc::new(
            VecModel::from(
                conversation.lock().unwrap().get_messages()
                    .into_iter()
                    .map(|msg| msg.into())
                    .collect::<Vec<slint_generatedAppWindow::SlintMessage>>()
            )
        );
        // ui.invoke_update_frontend_chat(messages_rc.clone());
        ui.set_messages(messages_rc.clone());
        
        tokens.lock().unwrap().store_history();
        
        // Update the frontend immediately
        // ui.invoke_update_frontend_chat(messages_rc.clone());
        ui.invoke_update_send_button_enabled(true)

    } else {
        eprintln!("Failed to upgrade weak pointer to AppWindow");
    }
}

fn main() -> Result<(), Error> {
    // prep
    let max_additional_tokens: i16 = 256;

    let tokenizer_path: &str = "./tokenisers/tokenizer.json";

    // let model_path: &str = "C:/Users/JamesJoslin/dev/rust-dev/james_ai/network/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/qwen2-0-5.onnx";

    let model_path: &str = "./qwen2-0-5.onnx"; 

    // let environment: Arc<Environment> = Arc::new(Environment::builder()
    //     .with_name("qwen2-0-5B")
    //     .with_log_level(LoggingLevel::Verbose)
    //     .build()?);

    let tokenizer_wrapper: Arc<Mutex<TokenizerWrapper>>  = Arc::new(Mutex::new(TokenizerWrapper::new(tokenizer_path).unwrap()));

    // let mut model_wrapper: Arc<Mutex<ModelWrapper>> = Arc::new(Mutex::new(ModelWrapper::new(
    //     model_path, &environment.lock().unwrap()).unwrap()));

    let model_wrapper: Arc<Mutex<ModelWrapper>> = Arc::new(
        Mutex::new(
            ModelWrapper::new(
                model_path.to_string(),
                // Arc::clone(&environment))
                // .unwrap()
            ).unwrap()
        )
    );

    let conversation_log:Arc<Mutex<Conversation>> = Arc::new(Mutex::new(Conversation::new()));

    let llm_model: LlmModel = LlmModel { inference: model_wrapper, tokens: tokenizer_wrapper, conversation: conversation_log };

    let ui: AppWindow = slint_generatedAppWindow::AppWindow::new()?;

    // Pass llm_model to the callback
    let ui_weak = ui.as_weak();
    ui.on_send_chat_to_backend(move |text| {
        let ui_weak_clone = ui_weak.clone();
        let llm_model: LlmModel = llm_model.clone();
        handle_send_chat_to_backend(ui_weak_clone, llm_model, text.to_owned().into(), max_additional_tokens);
        true
    });

    ui.run().unwrap();

    Ok(())
}


// let mut conversation_log: Conversation = Conversation::new();

// let ui_clone = ui.clone_strong();
// let model_clone = Arc::clone(&model);

// ui_clone.on_send_chat_to_backend(move |text| {
//     println!("TextEdit contains: {}", text);

//     conversation_log.add_user_prompt(text.to_string());

//     let mut response_string: String = "".to_string();
//     conversation_log.add_ai_response(&response_string);

//     tokenizer_wrapper.store_initial_tokenize(text.to_string().trim());
//     for _ in 0..max_additional_tokens {
//         let messages_rc: ModelRc<SlintMessage> = ModelRc::new(
//             VecModel::from(
//                 conversation_log.get_messages()
//                     .into_iter()
//                     .map(|msg| msg.into())
//                     .collect::<Vec<slint_generatedAppWindow::SlintMessage>>()
//             )
//         );
//         ui.invoke_update_frontend_chat(messages_rc);

//         let input_array = Array::from_shape_vec((1, tokenizer_wrapper.get_tokens_for_model().len()), tokenizer_wrapper.get_tokens_for_model().clone()).unwrap();
//         let session_lock = model_clone.session.lock().unwrap();
//         let outputs: Vec<OrtOwnedTensor<f32, ndarray::Dim<ndarray::IxDynImpl>>> = session_lock.run(vec![input_array]).unwrap();
//         let prediction: i64 = i64::try_from(
//             QuantileExt::argmax(
//                 &outputs[0]
//                     .slice(s![.., -1, ..])
//                     .into_shape([151936])
//                     .unwrap()
//             ).context("Argmax failed").unwrap()
//         ).expect("Conversion to i64 failed");

//         tokenizer_wrapper.append_token(prediction);

//         if tokenizer_wrapper.check_exit(prediction) {
//             break;
//         }

//         let decoded_token: String = tokenizer_wrapper.decode_token(prediction).unwrap();
//         response_string = format!("{}{}", response_string, decoded_token);

//         conversation_log.update_ai_response(decoded_token.clone());

//         print!("{}", decoded_token);
//         io::stdout().flush().unwrap();
//     }
//     tokenizer_wrapper.store_history();
//     true
// });


// Create a channel to communicate with the main thread
// let (tx, rx) = mpsc::channel();

// // Offload heavy computation to a new thread
// thread::spawn(move || {
//     let ten_millis = time::Duration::from_millis(100);
//     thread::sleep(ten_millis); // Simulate heavy computation
//     // println!("Hello");
    
//     // let mut tokens = tokens.lock().unwrap();
//     // let mut inference = inference.lock().
//     // tokens.store_initial_tokenize(&text);

    
//     // Notify the main thread that computation is done
//     tx.send(()).unwrap(); 
// });

// Handle the UI update on the main thread after computation completes
// let ui_weak_clone = ui_weak.clone();
// thread::spawn(move || {
//     // Wait for the background computation to complete
//     rx.recv().unwrap();
//     println!("hello-post-rx");

//     // Use upgrade_in_event_loop to ensure the upgrade happens on the correct thread
//     ui_weak_clone.upgrade_in_event_loop(|ui| {
//         println!("ui_weak upgraded successfully");
//         // Re-enable the send button after computation
//         ui.invoke_update_send_button_enabled(true);
//     }).unwrap_or_else(|_| {
//         println!("ui_weak upgrade failed");
//     });
// });