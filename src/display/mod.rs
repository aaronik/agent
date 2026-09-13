pub mod terminal;

pub use terminal::TerminalDisplay;

// Cursor-position queries consume terminal replies through crossterm's event
// reader. Never race those queries with the working-input poll/read pair.
pub(crate) static TERMINAL_INPUT: std::sync::Mutex<()> = std::sync::Mutex::new(());
