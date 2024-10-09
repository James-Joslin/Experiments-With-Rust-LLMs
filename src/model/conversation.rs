use serde::Deserialize;
use serde::Serialize;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Conversation {
    messages: Vec<Message>
}

impl Conversation {
    pub fn new() -> Conversation {
        Conversation {
            messages: Vec::new(),
            
        }
    }
    
    pub fn add_user_prompt(&mut self, text: String) {
        // let words: Vec<&str> = text.split("\nAnswer").collect();
        // let [first, second] = words.as_slice()  else { unreachable!() };
        self.messages.push(Message { is_user: true, text: text.trim().to_string() });
        // println!("{:#?}", self.messages)
    }

    pub fn add_ai_response(&mut self, text: &String) {
        self.messages.push(Message { is_user: false, text: text.to_string() });
    }

    pub fn update_ai_response(&mut self, new_text: String) {
        if let Some(last_message) = self.messages.last_mut() {
            if !last_message.is_user {
                last_message.text.push_str(&new_text);
            }
        }
    }

    pub fn get_messages(&self) -> Vec<Message> {
        self.messages.clone()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    pub is_user: bool,
    pub text: String,
}