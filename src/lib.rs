#[macro_export]
macro_rules! debug_log {
	($($arg:tt)*) => {{
		#[cfg(debug_assertions)]
		{
			eprintln!($($arg)*);
		}
	}};
}

pub mod act;
pub mod chronos;
pub mod config;
pub mod grammar_manager;
pub mod inspector;
pub mod mapper;
pub mod memory;
pub mod rules;
pub mod scanner;
pub mod server;
pub mod slicer;
pub mod universal;
pub mod vector_store;
pub mod workspace;
pub mod xml_builder;
